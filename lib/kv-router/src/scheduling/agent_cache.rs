// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::protocols::WorkerWithDpRank;
use serde::{Deserialize, Serialize};

const MAX_DIGEST_LEN: usize = 128;
const MAX_REUSABLE_TOKENS: usize = 4_000_000;
const MAX_KV_BYTES: u64 = 1 << 50;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentCacheTier {
    Device,
    HostPinned,
    Disk,
    External,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentCacheLifecycle {
    Active,
    ToolWaiting,
    Completed,
    Cancelled,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentCacheMissReason {
    Stale,
    EpochLag,
    WorkerOffline,
    PlannerOnly,
    ExactMiss,
    RestoreTimeout,
    InsufficientData,
    #[default]
    None,
}

/// Privacy-safe logical cache inventory exported by an inference worker.
/// Digests are opaque identities; raw tokens, prompts, queries, and block handles are forbidden.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCacheStatus {
    pub worker: WorkerWithDpRank,
    pub namespace_digest: String,
    pub identity_digest: String,
    pub reusable_tokens: usize,
    #[serde(default)]
    pub estimated_bytes: Option<u64>,
    #[serde(default)]
    pub actual_bytes: Option<u64>,
    #[serde(default)]
    pub tier: AgentCacheTier,
    pub epoch: u64,
    pub timestamp_ms: u64,
    #[serde(default)]
    pub physical_exact_hit: bool,
    #[serde(default)]
    pub planner_candidate: bool,
    #[serde(default)]
    pub lease_expires_at_ms: Option<u64>,
    #[serde(default)]
    pub lifecycle: AgentCacheLifecycle,
    #[serde(default)]
    pub miss_reason: AgentCacheMissReason,
    #[serde(default)]
    pub lookup_cost_blocks: Option<f64>,
    #[serde(default)]
    pub restore_cost_blocks: Option<f64>,
    #[serde(default)]
    pub transfer_bytes_per_block_cost: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentCacheRoutingSignals {
    pub namespace_digest: String,
    pub identity_digest: String,
    #[serde(default)]
    pub expected_epoch: u64,
    /// Producer time used for deterministic freshness checks; the router never trusts wall-clock
    /// values hidden in metadata.
    pub now_ms: u64,
    #[serde(default)]
    pub soft_affinity_worker: Option<WorkerWithDpRank>,
    #[serde(default)]
    pub statuses: Vec<AgentCacheStatus>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheAction {
    LocalExactHit,
    RemoteRestore,
    Recompute,
}

impl CacheAction {
    pub fn metric_label(self) -> &'static str {
        match self {
            Self::LocalExactHit => "local_exact_hit",
            Self::RemoteRestore => "remote_restore",
            Self::Recompute => "recompute",
        }
    }
}

impl AgentCacheTier {
    pub fn metric_label(self) -> &'static str {
        match self {
            Self::Device => "device",
            Self::HostPinned => "host_pinned",
            Self::Disk => "disk",
            Self::External => "external",
            Self::Unknown => "unknown",
        }
    }
}

impl AgentCacheMissReason {
    pub fn metric_label(self) -> &'static str {
        match self {
            Self::Stale => "stale",
            Self::EpochLag => "epoch_lag",
            Self::WorkerOffline => "worker_offline",
            Self::PlannerOnly => "planner_only",
            Self::ExactMiss => "exact_miss",
            Self::RestoreTimeout => "restore_timeout",
            Self::InsufficientData => "insufficient_data",
            Self::None => "selected",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CacheActionCost {
    pub action: CacheAction,
    pub saved_prefill_blocks: f64,
    pub queue_blocks: f64,
    pub lookup_blocks: f64,
    pub transfer_blocks: f64,
    pub restore_blocks: f64,
    pub load_imbalance_blocks: f64,
    pub staleness_risk_blocks: f64,
    pub affinity_break_blocks: f64,
    pub total_blocks: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AgentCacheDecision {
    pub action: CacheAction,
    pub tier: AgentCacheTier,
    /// A bounded delta added to the existing Dynamo score. Negative is better.
    pub score_adjustment: f64,
    pub cost: CacheActionCost,
    pub fallback_reason: AgentCacheMissReason,
}

impl AgentCacheRoutingSignals {
    pub fn is_valid(&self) -> bool {
        valid_digest(&self.namespace_digest)
            && valid_digest(&self.identity_digest)
            && self.statuses.len() <= 256
    }

    /// Compare local reuse, remote restore, and ordinary prefill without replacing Dynamo's
    /// queue/load score. Missing or untrusted data yields a zero adjustment.
    pub fn decision_for(
        &self,
        worker: WorkerWithDpRank,
        block_size: u32,
        inventory_ttl_ms: u64,
        max_adjustment_blocks: f64,
    ) -> AgentCacheDecision {
        self.decision_for_online(
            worker,
            block_size,
            inventory_ttl_ms,
            max_adjustment_blocks,
            |_| true,
        )
    }

    /// Variant used by the scheduler to reject inventory from workers that are no longer
    /// registered. Keeping liveness outside the wire payload prevents a stale producer from
    /// declaring itself online.
    pub fn decision_for_online<F>(
        &self,
        worker: WorkerWithDpRank,
        block_size: u32,
        inventory_ttl_ms: u64,
        max_adjustment_blocks: f64,
        is_online: F,
    ) -> AgentCacheDecision
    where
        F: Fn(WorkerWithDpRank) -> bool,
    {
        let recompute = CacheActionCost::recompute();
        if !self.is_valid() || block_size == 0 || !max_adjustment_blocks.is_finite() {
            return AgentCacheDecision::fallback(recompute, AgentCacheMissReason::InsufficientData);
        }

        let mut fallback_reason = AgentCacheMissReason::InsufficientData;
        let mut best: Option<(CacheActionCost, AgentCacheTier)> = None;
        for status in &self.statuses {
            if !is_online(status.worker) {
                fallback_reason = AgentCacheMissReason::WorkerOffline;
                continue;
            }
            let Err(reason) = status.validate_for(self, inventory_ttl_ms) else {
                if status.worker == worker && status.tier == AgentCacheTier::Device {
                    best = Some((status.local_cost(block_size, self, worker), status.tier));
                    break;
                }
                if let Some(remote) = status.remote_cost(block_size, self, worker)
                    && best.is_none_or(|(current, _)| remote.total_blocks < current.total_blocks)
                {
                    best = Some((remote, status.tier));
                }
                continue;
            };
            fallback_reason = reason;
        }

        let Some((cost, tier)) = best else {
            return AgentCacheDecision::fallback(recompute, fallback_reason);
        };
        if cost.total_blocks >= recompute.total_blocks {
            return AgentCacheDecision {
                action: CacheAction::Recompute,
                tier: AgentCacheTier::Unknown,
                score_adjustment: 0.0,
                cost: recompute,
                fallback_reason: AgentCacheMissReason::None,
            };
        }

        AgentCacheDecision {
            action: cost.action,
            tier,
            score_adjustment: cost.total_blocks.clamp(
                -max_adjustment_blocks.max(0.0),
                max_adjustment_blocks.max(0.0),
            ),
            cost,
            fallback_reason: AgentCacheMissReason::None,
        }
    }
}

impl AgentCacheStatus {
    fn validate_for(
        &self,
        request: &AgentCacheRoutingSignals,
        inventory_ttl_ms: u64,
    ) -> Result<(), AgentCacheMissReason> {
        if !valid_digest(&self.namespace_digest)
            || !valid_digest(&self.identity_digest)
            || self.namespace_digest != request.namespace_digest
            || self.identity_digest != request.identity_digest
            || self.reusable_tokens == 0
            || self.reusable_tokens > MAX_REUSABLE_TOKENS
            || self.estimated_bytes.is_some_and(|v| v > MAX_KV_BYTES)
            || self.actual_bytes.is_some_and(|v| v > MAX_KV_BYTES)
            || !finite_non_negative(self.lookup_cost_blocks)
            || !finite_non_negative(self.restore_cost_blocks)
            || !finite_non_negative(self.transfer_bytes_per_block_cost)
        {
            return Err(AgentCacheMissReason::InsufficientData);
        }
        if self.epoch < request.expected_epoch {
            return Err(AgentCacheMissReason::EpochLag);
        }
        if request.now_ms.saturating_sub(self.timestamp_ms) > inventory_ttl_ms
            || self.timestamp_ms > request.now_ms.saturating_add(inventory_ttl_ms)
        {
            return Err(AgentCacheMissReason::Stale);
        }
        if !self.physical_exact_hit {
            return Err(if self.planner_candidate {
                AgentCacheMissReason::PlannerOnly
            } else {
                self.miss_reason
            });
        }
        Ok(())
    }

    fn lifecycle_risk(&self, now_ms: u64) -> f64 {
        match self.lifecycle {
            AgentCacheLifecycle::Active => 0.0,
            AgentCacheLifecycle::ToolWaiting
                if self
                    .lease_expires_at_ms
                    .is_some_and(|expiry| expiry > now_ms) =>
            {
                0.0
            }
            AgentCacheLifecycle::ToolWaiting => 1.0,
            AgentCacheLifecycle::Completed => 1.5,
            AgentCacheLifecycle::Cancelled => 2.0,
            AgentCacheLifecycle::Unknown => 0.5,
        }
    }

    fn affinity_break(&self, request: &AgentCacheRoutingSignals, worker: WorkerWithDpRank) -> f64 {
        let lease_valid = self
            .lease_expires_at_ms
            .is_some_and(|expiry| expiry > request.now_ms);
        if lease_valid
            && request
                .soft_affinity_worker
                .is_some_and(|affinity| affinity != worker)
        {
            1.0
        } else {
            0.0
        }
    }

    fn local_cost(
        &self,
        block_size: u32,
        request: &AgentCacheRoutingSignals,
        worker: WorkerWithDpRank,
    ) -> CacheActionCost {
        let saved = self.reusable_tokens as f64 / block_size as f64;
        CacheActionCost::new(
            CacheAction::LocalExactHit,
            saved,
            0.0,
            0.0,
            0.0,
            self.lifecycle_risk(request.now_ms),
            self.affinity_break(request, worker),
        )
    }

    fn remote_cost(
        &self,
        block_size: u32,
        request: &AgentCacheRoutingSignals,
        worker: WorkerWithDpRank,
    ) -> Option<CacheActionCost> {
        if self.worker == worker && self.tier == AgentCacheTier::Device {
            return None;
        }
        let bytes = self.actual_bytes.or(self.estimated_bytes)?;
        let transfer_rate = self.transfer_bytes_per_block_cost?;
        let lookup = self.lookup_cost_blocks?;
        let restore = self.restore_cost_blocks?;
        let transfer = bytes as f64 * transfer_rate;
        if !transfer.is_finite() {
            return None;
        }
        let saved = self.reusable_tokens as f64 / block_size as f64;
        Some(CacheActionCost::new(
            CacheAction::RemoteRestore,
            saved,
            lookup,
            transfer,
            restore,
            self.lifecycle_risk(request.now_ms) + 0.5,
            self.affinity_break(request, worker),
        ))
    }
}

impl CacheActionCost {
    fn recompute() -> Self {
        Self {
            action: CacheAction::Recompute,
            saved_prefill_blocks: 0.0,
            queue_blocks: 0.0,
            lookup_blocks: 0.0,
            transfer_blocks: 0.0,
            restore_blocks: 0.0,
            load_imbalance_blocks: 0.0,
            staleness_risk_blocks: 0.0,
            affinity_break_blocks: 0.0,
            total_blocks: 0.0,
        }
    }

    fn new(
        action: CacheAction,
        saved_prefill_blocks: f64,
        lookup_blocks: f64,
        transfer_blocks: f64,
        restore_blocks: f64,
        staleness_risk_blocks: f64,
        affinity_break_blocks: f64,
    ) -> Self {
        let total_blocks = lookup_blocks
            + transfer_blocks
            + restore_blocks
            + staleness_risk_blocks
            + affinity_break_blocks
            - saved_prefill_blocks;
        Self {
            action,
            saved_prefill_blocks,
            queue_blocks: 0.0,
            lookup_blocks,
            transfer_blocks,
            restore_blocks,
            load_imbalance_blocks: 0.0,
            staleness_risk_blocks,
            affinity_break_blocks,
            total_blocks,
        }
    }
}

impl AgentCacheDecision {
    fn fallback(cost: CacheActionCost, fallback_reason: AgentCacheMissReason) -> Self {
        Self {
            action: CacheAction::Recompute,
            tier: AgentCacheTier::Unknown,
            score_adjustment: 0.0,
            cost,
            fallback_reason,
        }
    }
}

fn valid_digest(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_DIGEST_LEN
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
}

fn finite_non_negative(value: Option<f64>) -> bool {
    value.is_none_or(|v| v.is_finite() && v >= 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank {
            worker_id: id,
            dp_rank: 0,
        }
    }

    fn status(worker_id: u64) -> AgentCacheStatus {
        AgentCacheStatus {
            worker: worker(worker_id),
            namespace_digest: "ns1".into(),
            identity_digest: "id1".into(),
            reusable_tokens: 1024,
            estimated_bytes: Some(1024),
            actual_bytes: None,
            tier: AgentCacheTier::Device,
            epoch: 3,
            timestamp_ms: 1_000,
            physical_exact_hit: true,
            planner_candidate: true,
            lease_expires_at_ms: Some(2_000),
            lifecycle: AgentCacheLifecycle::Active,
            miss_reason: AgentCacheMissReason::None,
            lookup_cost_blocks: Some(1.0),
            restore_cost_blocks: Some(1.0),
            transfer_bytes_per_block_cost: Some(0.001),
        }
    }

    fn signals(statuses: Vec<AgentCacheStatus>) -> AgentCacheRoutingSignals {
        AgentCacheRoutingSignals {
            namespace_digest: "ns1".into(),
            identity_digest: "id1".into(),
            expected_epoch: 3,
            now_ms: 1_100,
            soft_affinity_worker: None,
            statuses,
        }
    }

    #[test]
    fn local_exact_hit_gets_bounded_credit() {
        let decision = signals(vec![status(1)]).decision_for(worker(1), 16, 500, 8.0);
        assert_eq!(decision.action, CacheAction::LocalExactHit);
        assert_eq!(decision.score_adjustment, -8.0);
    }

    #[test]
    fn remote_restore_must_beat_recompute() {
        let mut cheap = status(1);
        cheap.tier = AgentCacheTier::HostPinned;
        assert_eq!(
            signals(vec![cheap])
                .decision_for(worker(2), 16, 500, 16.0)
                .action,
            CacheAction::RemoteRestore
        );
        let mut costly = status(1);
        costly.tier = AgentCacheTier::HostPinned;
        costly.transfer_bytes_per_block_cost = Some(1.0);
        assert_eq!(
            signals(vec![costly])
                .decision_for(worker(2), 16, 500, 16.0)
                .action,
            CacheAction::Recompute
        );
    }

    #[test]
    fn planner_candidate_is_not_a_physical_hit() {
        let mut s = status(1);
        s.physical_exact_hit = false;
        let decision = signals(vec![s]).decision_for(worker(1), 16, 500, 8.0);
        assert_eq!(decision.action, CacheAction::Recompute);
        assert_eq!(decision.fallback_reason, AgentCacheMissReason::PlannerOnly);
    }

    #[test]
    fn execution_miss_and_restore_timeout_fall_back_to_recompute() {
        for reason in [
            AgentCacheMissReason::ExactMiss,
            AgentCacheMissReason::RestoreTimeout,
        ] {
            let mut unavailable = status(1);
            unavailable.physical_exact_hit = false;
            unavailable.planner_candidate = false;
            unavailable.miss_reason = reason;

            let decision = signals(vec![unavailable]).decision_for(worker(1), 16, 500, 8.0);
            assert_eq!(decision.action, CacheAction::Recompute);
            assert_eq!(decision.score_adjustment, 0.0);
            assert_eq!(decision.fallback_reason, reason);
        }
    }

    #[test]
    fn stale_epoch_and_missing_costs_fail_closed() {
        let mut stale = status(1);
        stale.timestamp_ms = 1;
        assert_eq!(
            signals(vec![stale])
                .decision_for(worker(1), 16, 10, 8.0)
                .fallback_reason,
            AgentCacheMissReason::Stale
        );
        let mut epoch = status(1);
        epoch.epoch = 2;
        assert_eq!(
            signals(vec![epoch])
                .decision_for(worker(1), 16, 500, 8.0)
                .fallback_reason,
            AgentCacheMissReason::EpochLag
        );
        let mut missing = status(1);
        missing.tier = AgentCacheTier::HostPinned;
        missing.actual_bytes = None;
        missing.estimated_bytes = None;
        assert_eq!(
            signals(vec![missing])
                .decision_for(worker(2), 16, 500, 8.0)
                .action,
            CacheAction::Recompute
        );
    }

    #[test]
    fn expired_lease_removes_affinity_credit_and_completed_decays() {
        let mut expired = status(1);
        expired.lifecycle = AgentCacheLifecycle::ToolWaiting;
        expired.lease_expires_at_ms = Some(1_050);
        let active = status(1);
        let expired_cost = signals(vec![expired])
            .decision_for(worker(1), 16, 500, 128.0)
            .cost
            .total_blocks;
        let active_cost = signals(vec![active])
            .decision_for(worker(1), 16, 500, 128.0)
            .cost
            .total_blocks;
        assert!(expired_cost > active_cost);
        let mut completed = status(1);
        completed.lifecycle = AgentCacheLifecycle::Completed;
        assert!(
            signals(vec![completed])
                .decision_for(worker(1), 16, 500, 128.0)
                .cost
                .total_blocks
                > active_cost
        );
    }

    #[test]
    fn offline_inventory_fails_closed() {
        let decision =
            signals(vec![status(1)]).decision_for_online(worker(2), 16, 500, 8.0, |_| false);
        assert_eq!(decision.action, CacheAction::Recompute);
        assert_eq!(
            decision.fallback_reason,
            AgentCacheMissReason::WorkerOffline
        );
        assert_eq!(decision.score_adjustment, 0.0);
    }

    #[test]
    fn metric_labels_are_fixed_low_cardinality_values() {
        assert_eq!(CacheAction::LocalExactHit.metric_label(), "local_exact_hit");
        assert_eq!(CacheAction::RemoteRestore.metric_label(), "remote_restore");
        assert_eq!(
            AgentCacheMissReason::PlannerOnly.metric_label(),
            "planner_only"
        );
        assert_eq!(AgentCacheMissReason::None.metric_label(), "selected");
        assert_eq!(AgentCacheTier::HostPinned.metric_label(), "host_pinned");
        assert_eq!(AgentCacheTier::Unknown.metric_label(), "unknown");
    }
}
