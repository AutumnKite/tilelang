/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file schedule_builder.cc
 * \brief ScheduleUnitBuilder and task gathering for TileLang AutoSchedule
 */

#include "./schedule_builder.h"

#include "../auto_schedule.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/target/target.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../op/builtin.h"
#include "../../op/gemm.h"
#include "../../op/utils.h"
#include "../../target/utils.h"
#include "../common/attr.h"
#include "../common/collector.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using ffi::GetRef;

void GatherTaskNodes(const std::vector<std::shared_ptr<IRStructure>> &nodes,
                     std::vector<std::shared_ptr<IRStructure>> &task_nodes) {
  for (const auto &node : nodes) {
    if (node->IsTask()) {
      task_nodes.emplace_back(node);
    } else if (node->IsSequence()) {
      auto seq = static_cast<SequenceNode *>(node.get());
      GatherTaskNodes(seq->children, task_nodes);
    } else if (node->IsWrapper()) {
      auto wrapper = static_cast<WrapperNode *>(node.get());
      if (wrapper->task)
        task_nodes.emplace_back(wrapper->task);
      if (wrapper->child)
        GatherTaskNodesSingle(wrapper->child, task_nodes);
    } else if (node->IsControl()) {
      task_nodes.emplace_back(node);
    } else if (node->IsIf()) {
      task_nodes.emplace_back(node);
    } else {
      LOG(FATAL) << "Unknown node type in GatherTaskNodes";
    }
  }
}

void GatherTaskNodesSingle(
    const std::shared_ptr<IRStructure> &node,
    std::vector<std::shared_ptr<IRStructure>> &task_nodes) {
  return GatherTaskNodes({node}, task_nodes);
}

bool SameBuffer(const BufferRegion &a, const BufferRegion &b) {
  return a->buffer.same_as(b->buffer);
}

bool SameVar(const Var &a, const Var &b) { return a.same_as(b); }

bool HasDependency(const IRStructure *a, const IRStructure *b) {
  if (a->ContainsLoopBreak())
    return true;
  if (b->ContainsLoopBreak())
    return true;
  for (const auto &write_region_a : a->GetWriteRegions()) {
    for (const auto &read_region_b : b->GetReadRegions()) {
      if (SameBuffer(write_region_a, read_region_b))
        return true;
    }
    for (const auto &write_region_b : b->GetWriteRegions()) {
      if (SameBuffer(write_region_a, write_region_b))
        return true;
    }
  }
  for (const auto &read_region_a : a->GetReadRegions()) {
    for (const auto &write_region_b : b->GetWriteRegions()) {
      if (SameBuffer(read_region_a, write_region_b))
        return true;
    }
  }
  for (const auto &write_var_a : a->GetWriteVars()) {
    for (const auto &read_var_b : b->GetReadVars()) {
      if (SameVar(write_var_a, read_var_b))
        return true;
    }
  }
  return false;
}

bool HasRegisterDependency(const IRStructure *a, const IRStructure *b) {
  if (a->ContainsLoopBreak())
    return true;
  if (b->ContainsLoopBreak())
    return true;
  for (const auto &write_region_a : a->GetWriteRegions()) {
    if (IsSharedBuffer(write_region_a.get()->buffer))
      continue;
    for (const auto &read_region_b : b->GetReadRegions()) {
      if (SameBuffer(write_region_a, read_region_b))
        return true;
    }
    for (const auto &write_region_b : b->GetWriteRegions()) {
      if (SameBuffer(write_region_a, write_region_b))
        return true;
    }
  }
  for (const auto &read_region_a : a->GetReadRegions()) {
    if (IsSharedBuffer(read_region_a.get()->buffer))
      continue;
    for (const auto &write_region_b : b->GetWriteRegions()) {
      if (SameBuffer(read_region_a, write_region_b))
        return true;
    }
  }
  return false;
}

std::set<Buffer> GetSharedDependencies(const IRStructure *a,
                                       const IRStructure *b) {
  std::set<Buffer> deps;
  for (const auto &write_region_a : a->GetWriteRegions()) {
    if (!IsSharedBuffer(write_region_a->buffer))
      continue;
    for (const auto &read_region_b : b->GetReadRegions()) {
      if (SameBuffer(write_region_a, read_region_b))
        deps.insert(write_region_a->buffer);
    }
    for (const auto &write_region_b : b->GetWriteRegions()) {
      if (SameBuffer(write_region_a, write_region_b))
        deps.insert(write_region_a->buffer);
    }
  }
  for (const auto &read_region_a : a->GetReadRegions()) {
    if (!IsSharedBuffer(read_region_a->buffer))
      continue;
    for (const auto &write_region_b : b->GetWriteRegions()) {
      if (SameBuffer(read_region_a, write_region_b))
        deps.insert(read_region_a->buffer);
    }
  }
  return deps;
}

bool HasRegisterRegion(const IRStructure *node) {
  return CountRegisterRegions(node) > 0;
}

// Collect register buffers read by all broadcast tasks in the IR tree.
static void CollectBroadcastRegisterReads(
    IRStructure *node, std::unordered_set<const BufferNode *> &reg_bufs) {
  if (!node)
    return;
  auto collect_from_leaf_task = [&](const TaskNode *task) {
    if (!task)
      return;
    int wg_id = task->GetWarpgroupId();
    if (!IsWarpgroupBroadcast(wg_id) && wg_id != kWarpgroupUnassigned)
      return;
    for (const auto &region : task->GetReadRegions()) {
      if (IsRegisterRegion(region)) {
        reg_bufs.insert(region->buffer.get());
      }
    }
  };
  auto collect_from_structural_task = [&](const TaskNode *task) {
    if (!task)
      return;
    for (const auto &region : task->GetReadRegions()) {
      if (IsRegisterRegion(region)) {
        reg_bufs.insert(region->buffer.get());
      }
    }
  };

  if (node->IsTask()) {
    collect_from_leaf_task(static_cast<TaskNode *>(node));
  } else if (node->IsControl()) {
    auto ctrl = static_cast<ControlNode *>(node);
    if (ctrl->task)
      collect_from_structural_task(ctrl->task.get());
    CollectBroadcastRegisterReads(ctrl->child.get(), reg_bufs);
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<WrapperNode *>(node);
    if (wrapper->task)
      collect_from_structural_task(wrapper->task.get());
    CollectBroadcastRegisterReads(wrapper->child.get(), reg_bufs);
  } else if (node->IsSequence()) {
    auto seq = static_cast<SequenceNode *>(node);
    for (auto &child : seq->children) {
      CollectBroadcastRegisterReads(child.get(), reg_bufs);
    }
  } else if (node->IsScheduleUnit()) {
    auto unit = static_cast<ScheduleUnit *>(node);
    CollectBroadcastRegisterReads(unit->child.get(), reg_bufs);
  } else if (node->IsIf()) {
    auto if_node = static_cast<IfNode *>(node);
    if (if_node->task)
      collect_from_structural_task(if_node->task.get());
    CollectBroadcastRegisterReads(if_node->then_child.get(), reg_bufs);
    if (if_node->else_child)
      CollectBroadcastRegisterReads(if_node->else_child.get(), reg_bufs);
  }
}

// Propagate broadcast: if a broadcast task reads a register buffer,
// any leaf task that writes that register buffer must also be broadcast
// (because each wg needs its own initialized copy).
void PropagateBroadcastWarpgroupId(IRStructure *root) {
  std::vector<TaskNodeWithContext> all_tasks;
  CollectAllTaskNodesWithContext(root, all_tasks);

  bool changed = true;
  while (changed) {
    changed = false;
    // 1. Collect register buffers read by all current broadcast tasks
    std::unordered_set<const BufferNode *> broadcast_reg_reads;
    CollectBroadcastRegisterReads(root, broadcast_reg_reads);
    // Also collect from leaf broadcast tasks
    for (auto &task_ctx : all_tasks) {
      TaskNode *task = task_ctx.task;
      if (!IsWarpgroupBroadcast(task->GetWarpgroupId()))
        continue;
      for (const auto &region : task->GetReadRegions()) {
        if (IsRegisterRegion(region)) {
          broadcast_reg_reads.insert(region->buffer.get());
        }
      }
    }
    // 2. Mark leaf tasks that write these register buffers as broadcast
    if (!broadcast_reg_reads.empty()) {
      for (auto &task_ctx : all_tasks) {
        TaskNode *task = task_ctx.task;
        if (IsWarpgroupBroadcast(task->GetWarpgroupId()))
          continue;
        for (const auto &region : task->GetWriteRegions()) {
          if (IsRegisterRegion(region) &&
              broadcast_reg_reads.count(region->buffer.get())) {
            task->SetWarpgroupId(kWarpgroupBroadcast);
            changed = true;
            break;
          }
        }
      }
    }

    // 3. Detect cross-warpgroup register buffer / scalar-var accesses
    constexpr int kReaderAnyWg = std::numeric_limits<int>::min();
    std::unordered_map<const BufferNode *, std::unordered_set<int>>
        buffer_reader_wgs;
    std::unordered_map<const BufferNode *, std::unordered_set<int>>
        buffer_writer_wgs;
    std::unordered_map<const VarNode *, std::unordered_set<int>> var_reader_wgs;
    std::unordered_map<const VarNode *, std::unordered_set<int>> var_writer_wgs;

    auto add_accesses = [&](const TaskNode *task, int reader_key, int wg_id) {
      for (const auto &region : task->GetReadRegions()) {
        if (IsRegisterRegion(region)) {
          buffer_reader_wgs[region->buffer.get()].insert(reader_key);
        }
      }
      if (wg_id >= 0) {
        for (const auto &region : task->GetWriteRegions()) {
          if (IsRegisterRegion(region)) {
            buffer_writer_wgs[region->buffer.get()].insert(wg_id);
          }
        }
      }
      for (const auto &v : task->GetReadVars()) {
        var_reader_wgs[v.get()].insert(reader_key);
      }
      if (wg_id >= 0) {
        for (const auto &v : task->GetWriteVars()) {
          var_writer_wgs[v.get()].insert(wg_id);
        }
      }
    };

    for (auto &task_ctx : all_tasks) {
      TaskNode *task = task_ctx.task;
      int wg_id = task->GetWarpgroupId();
      int reader_key = (wg_id >= 0) ? wg_id : kReaderAnyWg;
      add_accesses(task, reader_key, wg_id);
    }
    // Also include structural tasks
    std::function<void(IRStructure *)> walk_structural =
        [&](IRStructure *node) {
          if (!node)
            return;
          auto add_struct = [&](const TaskNode *t) {
            if (!t)
              return;
            int wg = t->GetWarpgroupId();
            int key = (wg >= 0) ? wg : kReaderAnyWg;
            if (IsWarpgroupBroadcast(wg))
              key = kReaderAnyWg;
            add_accesses(t, key, -1);
          };
          if (node->IsControl()) {
            auto *c = static_cast<ControlNode *>(node);
            add_struct(c->task.get());
            walk_structural(c->child.get());
          } else if (node->IsWrapper()) {
            auto *w = static_cast<WrapperNode *>(node);
            add_struct(w->task.get());
            walk_structural(w->child.get());
          } else if (node->IsIf()) {
            auto *i = static_cast<IfNode *>(node);
            add_struct(i->task.get());
            walk_structural(i->then_child.get());
            walk_structural(i->else_child.get());
          } else if (node->IsSequence()) {
            auto *s = static_cast<SequenceNode *>(node);
            for (auto &c : s->children)
              walk_structural(c.get());
          } else if (node->IsScheduleUnit()) {
            auto *u = static_cast<ScheduleUnit *>(node);
            walk_structural(u->child.get());
          }
          // Leaf tasks already handled by the all_tasks loop.
        };
    walk_structural(root);

    std::unordered_set<const BufferNode *> cross_wg_buffers;
    for (const auto &kv : buffer_reader_wgs) {
      const auto *buf = kv.first;
      const auto &reader_wgs = kv.second;
      const auto &writer_wgs = buffer_writer_wgs[buf];
      for (int rwg : reader_wgs) {
        if (writer_wgs.find(rwg) == writer_wgs.end()) {
          cross_wg_buffers.insert(buf);
          break;
        }
      }
    }

    std::unordered_set<const VarNode *> cross_wg_vars;
    for (const auto &kv : var_reader_wgs) {
      const auto *v = kv.first;
      const auto &reader_wgs = kv.second;
      auto it_w = var_writer_wgs.find(v);
      if (it_w == var_writer_wgs.end())
        continue;
      const auto &writer_wgs = it_w->second;
      for (int rwg : reader_wgs) {
        if (writer_wgs.find(rwg) == writer_wgs.end()) {
          cross_wg_vars.insert(v);
          break;
        }
      }
    }

    if (!cross_wg_buffers.empty() || !cross_wg_vars.empty()) {
      for (auto &task_ctx : all_tasks) {
        TaskNode *task = task_ctx.task;
        if (IsWarpgroupBroadcast(task->GetWarpgroupId()))
          continue;
        bool should_broadcast = false;
        for (const auto &region : task->GetWriteRegions()) {
          if (IsRegisterRegion(region) &&
              cross_wg_buffers.count(region->buffer.get())) {
            should_broadcast = true;
            break;
          }
        }
        if (!should_broadcast) {
          for (const auto &v : task->GetWriteVars()) {
            if (cross_wg_vars.count(v.get())) {
              should_broadcast = true;
              break;
            }
          }
        }
        if (should_broadcast) {
          task->SetWarpgroupId(kWarpgroupBroadcast);
          changed = true;
        }
      }
    }
  }
}

bool HasResourceDependency(const IRStructure *a, const IRStructure *b) {
  if (a->UsesTMACore() && b->UsesTMACore())
    return true;
  if (a->UsesTensorCore() && b->UsesTensorCore())
    return true;
  if (a->UsesCUDACore() && b->UsesCUDACore())
    return true;
  return false;
}

void CollectPrefixTasks(IRStructure *root,
                        std::unordered_set<TaskNode *> &prefix_tasks) {
  if (!root)
    return;

  // Collect all top-level items (TaskNodes and ControlNodes) in order
  std::vector<IRStructure *> items;
  CollectTopLevelItems(root, items);

  // Walk forward: a task is prefix iff it has no register regions AND
  // has no dependency with any previously-rejected (non-prefix) item.
  // ControlNodes are always rejected.
  std::vector<IRStructure *> rejected;
  for (auto *item : items) {
    if (item->IsControl()) {
      rejected.push_back(item);
      continue;
    }
    if (!item->IsTask()) {
      rejected.push_back(item);
      continue;
    }
    auto *task = static_cast<TaskNode *>(item);
    if (CountRegisterRegions(task) != 0) {
      rejected.push_back(task);
      continue;
    }
    bool has_dep = false;
    for (auto *rej : rejected) {
      if (HasDependency(task, rej)) {
        has_dep = true;
        break;
      }
    }
    for (auto *pre : prefix_tasks) {
      if (HasDependency(task, pre)) {
        has_dep = true;
        break;
      }
    }
    if (has_dep) {
      rejected.push_back(task);
    } else {
      prefix_tasks.insert(task);
    }
  }
}

void CollectSuffixTasks(IRStructure *root,
                        const std::vector<TaskNodeWithContext> &all_tasks,
                        const TaskUnionFind &uf,
                        std::unordered_set<TaskNode *> &suffix_tasks) {
  if (!root)
    return;

  // Collect all top-level items in order
  std::vector<IRStructure *> items;
  CollectTopLevelItems(root, items);

  // Walk backward: a task is a suffix candidate iff it has no dependency
  // with any subsequently-rejected (non-suffix) item.
  // ControlNodes are always rejected.
  std::vector<IRStructure *> rejected;
  std::unordered_set<TaskNode *> candidate_set;
  for (int i = static_cast<int>(items.size()) - 1; i >= 0; --i) {
    auto *item = items[i];
    if (item->GetSchedulePhase() == SchedulePhase::kPrologue) {
      rejected.push_back(item);
      continue;
    }
    if (item->IsControl()) {
      rejected.push_back(item);
      continue;
    }
    if (!item->IsTask()) {
      rejected.push_back(item);
      continue;
    }
    auto *task = static_cast<TaskNode *>(item);
    if (task->ContainsLoopBreak()) {
      rejected.push_back(task);
      continue;
    }
    bool has_dep = false;
    for (auto *rej : rejected) {
      if (HasDependency(task, rej)) {
        has_dep = true;
        break;
      }
    }
    if (has_dep) {
      rejected.push_back(task);
    } else {
      candidate_set.insert(task);
    }
  }

  // Apply register region grouping via union-find
  std::unordered_map<TaskNode *, int> task_to_index;
  task_to_index.reserve(all_tasks.size());
  for (int i = 0; i < static_cast<int>(all_tasks.size()); ++i) {
    task_to_index[all_tasks[i].task] = i;
  }

  std::unordered_map<int, std::vector<TaskNode *>> component_tasks;
  component_tasks.reserve(all_tasks.size());
  for (int i = 0; i < static_cast<int>(all_tasks.size()); ++i) {
    int root_idx = uf.find(i);
    component_tasks[root_idx].push_back(all_tasks[i].task);
  }

  for (TaskNode *task : candidate_set) {
    if (suffix_tasks.count(task))
      continue;

    if (CountRegisterRegions(task) == 0) {
      suffix_tasks.insert(task);
      continue;
    }

    auto it = task_to_index.find(task);
    if (it == task_to_index.end())
      continue;
    int component_root = uf.find(it->second);

    auto comp_it = component_tasks.find(component_root);
    if (comp_it == component_tasks.end())
      continue;

    bool all_in_candidates = true;
    for (TaskNode *component_task : comp_it->second) {
      if (candidate_set.find(component_task) == candidate_set.end()) {
        all_in_candidates = false;
        break;
      }
    }

    if (!all_in_candidates)
      continue;

    for (TaskNode *component_task : comp_it->second) {
      suffix_tasks.insert(component_task);
    }
  }
}

std::vector<PrimExpr>
AssignWarpgroupIdsGlobal(IRStructure *root, const WarpSpecializeConfig &config,
                         PrimExpr thread_count) {
  if (!root) {
    LOG(FATAL) << "Empty root";
  }

  std::vector<TaskNodeWithContext> all_tasks;
  CollectAllTaskNodesWithContext(root, all_tasks);

  if (all_tasks.empty()) {
    LOG(FATAL) << "No task";
  }

  int n = all_tasks.size();

  for (auto &task_ctx : all_tasks) {
    task_ctx.task->SetWarpgroupId(kWarpgroupUnassigned);
  }

  // Tasks with loop_break are broadcast to all warp groups
  for (auto &task_ctx : all_tasks) {
    if (task_ctx.task->ContainsLoopBreak()) {
      task_ctx.task->SetWarpgroupId(kWarpgroupBroadcast);
    }
  }

  TaskUnionFind uf(n);
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      if (UseSameRegisterRegion(all_tasks[i].task, all_tasks[j].task)) {
        uf.unite(i, j);
      }
    }
  }

  std::unordered_set<TaskNode *> prefix_tasks;
  CollectPrefixTasks(root, prefix_tasks);
  for (auto *task : prefix_tasks) {
    task->SetSchedulePhase(SchedulePhase::kPrologue);
    task->SetWarpgroupId(0);
  }

  std::unordered_set<TaskNode *> suffix_tasks;
  CollectSuffixTasks(root, all_tasks, uf, suffix_tasks);
  for (auto *task : suffix_tasks) {
    task->SetSchedulePhase(SchedulePhase::kEpilogue);
    task->SetWarpgroupId(0);
  }

  std::unordered_map<int, std::vector<int>> components;
  for (int i = 0; i < n; i++) {
    int root_idx = uf.find(i);
    components[root_idx].push_back(i);
  }

  std::vector<ComponentInfo> component_infos;
  for (const auto &kv : components) {
    int root = kv.first;
    const std::vector<int> &indices = kv.second;
    bool has_register_region = false;
    for (int idx : indices) {
      if (CountRegisterRegions(all_tasks[idx].task) > 0) {
        has_register_region = true;
        break;
      }
    }
    int64_t total_weighted_latency = 0;
    bool has_task = false;
    bool has_tma_core = false;
    bool has_tensor_core = false;
    for (int idx : indices) {
      TaskNode *task = all_tasks[idx].task;
      if (prefix_tasks.find(task) != prefix_tasks.end()) {
        continue;
      }
      if (suffix_tasks.find(task) != suffix_tasks.end()) {
        continue;
      }
      if (task->ContainsLoopBreak()) {
        continue;
      }
      has_task = true;
      int64_t latency = task->GetLatency();
      int64_t tripcount = all_tasks[idx].tripcount;
      total_weighted_latency += latency * tripcount;
      has_tma_core |= task->UsesTMACore();
      has_tensor_core |= task->UsesTensorCore();
    }
    if (has_task) {
      component_infos.push_back({root, total_weighted_latency, indices,
                                 has_tma_core, has_tensor_core});
    }
  }

  std::sort(component_infos.begin(), component_infos.end(),
            [](const ComponentInfo &a, const ComponentInfo &b) {
              return a.weighted_latency > b.weighted_latency;
            });

  if (config.enable_warp_partition) {
    for (const auto &comp : component_infos) {
      int assigned_warpgroup = 0;
      if (comp.uses_tensor_core_ && !comp.uses_tma_core_) {
        assigned_warpgroup = 0;
      } else if (!comp.uses_tensor_core_ && comp.uses_tma_core_) {
        assigned_warpgroup = 1;
      } else {
        assigned_warpgroup = 3;
      }
      for (int idx : comp.task_indices) {
        TaskNode *task = all_tasks[idx].task;
        if (!task->ContainsLoopBreak()) {
          task->SetWarpgroupId(assigned_warpgroup);
        }
      }
    }
    return {IntImm(DataType::Int(32), 32), IntImm(DataType::Int(32), 32),
            IntImm(DataType::Int(32), 64),
            thread_count - IntImm(DataType::Int(32), 128)};
  }

  int64_t warpgroup0_latency = 0;
  int64_t warpgroup1_latency = 0;

  for (const auto &comp : component_infos) {
    int assigned_warpgroup = 0;
    if (warpgroup0_latency <= warpgroup1_latency) {
      assigned_warpgroup = 0;
      warpgroup0_latency += comp.weighted_latency;
    } else {
      assigned_warpgroup = 1;
      warpgroup1_latency += comp.weighted_latency;
    }
  }

  int64_t max_latency = std::max(warpgroup0_latency, warpgroup1_latency);
  int64_t min_latency = std::min(warpgroup0_latency, warpgroup1_latency);
  if ((double)max_latency < 1.1 * min_latency) {
    int64_t warpgroup0_latency = 0;
    int64_t warpgroup1_latency = 0;

    for (const auto &comp : component_infos) {
      int assigned_warpgroup = 0;
      if (warpgroup0_latency <= warpgroup1_latency) {
        assigned_warpgroup = 0;
        warpgroup0_latency += comp.weighted_latency;
      } else {
        assigned_warpgroup = 1;
        warpgroup1_latency += comp.weighted_latency;
      }

      for (int idx : comp.task_indices) {
        TaskNode *task = all_tasks[idx].task;
        if (!task->ContainsLoopBreak()) {
          task->SetWarpgroupId(assigned_warpgroup);
        }
      }
    }
    return {thread_count, thread_count};
  } else {
    int64_t warpgroup0_latency = 0;
    int64_t warpgroup1_latency = 0;
    for (const auto &comp : component_infos) {
      int assigned_warpgroup = 0;
      if (comp.uses_tensor_core_ && !comp.uses_tma_core_) {
        assigned_warpgroup = 0;
        warpgroup0_latency += comp.weighted_latency;
      } else if (!comp.uses_tensor_core_ && comp.uses_tma_core_) {
        assigned_warpgroup = 1;
        warpgroup1_latency += comp.weighted_latency;
      } else if (warpgroup0_latency <= warpgroup1_latency) {
        assigned_warpgroup = 0;
        warpgroup0_latency += comp.weighted_latency;
      } else {
        assigned_warpgroup = 1;
        warpgroup1_latency += comp.weighted_latency;
      }

      for (int idx : comp.task_indices) {
        TaskNode *task = all_tasks[idx].task;
        if (!task->ContainsLoopBreak()) {
          task->SetWarpgroupId(assigned_warpgroup);
        }
      }
    }
    return {thread_count,
            IntImm(DataType::Int(32), config.producer_thread_count)};
  }
}

/*
  Recursively schedule root node, after scheduling IRStructure satisfies the
following properties: 1) For each SequenceNode, its children are reordered by Z3
scheduler and wrapped in ScheduleUnits. 2) For each ControlNode, its child is a
SequenceNode with Z3-scheduled children wrapped in ScheduleUnits. 3) For each
IfNode, its then_child and else_child are recursively scheduled (if exist).
*/
void ScheduleUnitBuilder::ScheduleRecursive(
    std::shared_ptr<IRStructure> &node, const std::set<Buffer> &used_buffers) {
  if (!node)
    return;

  auto ChildrenScheduleHelper =
      [&](std::vector<std::shared_ptr<IRStructure>> origin_children)
      -> std::vector<std::shared_ptr<IRStructure>> {
    std::vector<IRStructure *> child_nodes;
    child_nodes.reserve(origin_children.size());
    for (const auto &child : origin_children) {
      child_nodes.push_back(child.get());
    }

    std::vector<IRStructure *> sorted_nodes;
    sorted_nodes = Z3SchedulePython(child_nodes);

    bool order_changed = false;
    for (size_t i = 0; i < sorted_nodes.size(); ++i) {
      if (sorted_nodes[i] != child_nodes[i]) {
        order_changed = true;
        break;
      }
    }

    if (order_changed) {
      std::unordered_map<IRStructure *, size_t> node_to_index;
      for (size_t i = 0; i < child_nodes.size(); ++i) {
        node_to_index[child_nodes[i]] = i;
      }

      std::vector<std::shared_ptr<IRStructure>> reordered_children;
      reordered_children.reserve(sorted_nodes.size());

      for (IRStructure *sorted_node : sorted_nodes) {
        auto it = node_to_index.find(sorted_node);
        if (it == node_to_index.end()) {
          LOG(FATAL) << "[ScheduleRecursive] IRStructure not found in "
                        "children mapping";
        }
        size_t old_idx = it->second;
        reordered_children.emplace_back(origin_children[old_idx]);
      }

      origin_children = reordered_children;
    }
    for (auto &node : origin_children) {
      auto unit = std::make_shared<ScheduleUnit>();
      unit->stage = -1;
      unit->child = std::shared_ptr<IRStructure>(node);
      node = unit;
    }
    return origin_children;
  };

  if (node->IsTask()) {
    return;
  } else if (node->IsSequence()) {
    auto seq = static_cast<SequenceNode *>(node.get());

    std::vector<std::shared_ptr<IRStructure>> seq_children, origin_children;
    GatherTaskNodes(seq->children, origin_children);
    for (auto &child : origin_children) {
      auto child_used_buffers = used_buffers;
      for (auto &other_child : origin_children) {
        if (child.get() != other_child.get()) {
          for (const auto &region : other_child->GetReadRegions()) {
            child_used_buffers.insert(region.get()->buffer);
          }
          for (const auto &region : other_child->GetWriteRegions()) {
            child_used_buffers.insert(region.get()->buffer);
          }
        }
      }
      ScheduleRecursive(child, child_used_buffers);
    }

    seq->children = ChildrenScheduleHelper(origin_children);
    int64_t overall_latency = 0;
    for (const auto &child : seq->children) {
      overall_latency += child->GetLatency();
    }
    seq->SetLatency(overall_latency);
    seq->SetII(overall_latency);
    return;
  } else if (node->IsControl()) {
    auto ctrl = static_cast<ControlNode *>(node.get());

    if (ctrl->child) {
      if (ctrl->child->IsSequence()) {
        auto seq_body = static_cast<SequenceNode *>(ctrl->child.get());
        std::vector<std::shared_ptr<IRStructure>> origin_children;
        GatherTaskNodes(seq_body->children, origin_children);
        for (auto &child : origin_children) {
          auto child_used_buffers = used_buffers;
          for (auto &other_child : origin_children) {
            if (child.get() != other_child.get()) {
              for (const auto &region : other_child->GetReadRegions()) {
                child_used_buffers.insert(region.get()->buffer);
              }
              for (const auto &region : other_child->GetWriteRegions()) {
                child_used_buffers.insert(region.get()->buffer);
              }
            }
          }
          ScheduleRecursive(child, child_used_buffers);
        }
        Z3SchedulePythonLoop(ctrl, used_buffers);
      } else if (ctrl->child->IsWrapper()) {
        auto wrapper = static_cast<WrapperNode *>(ctrl->child.get());
        std::vector<std::shared_ptr<IRStructure>> origin_children;
        GatherTaskNodes({wrapper->task, wrapper->child}, origin_children);
        for (auto &child : origin_children) {
          auto child_used_buffers = used_buffers;
          for (auto &other_child : origin_children) {
            if (child.get() != other_child.get()) {
              for (const auto &region : other_child->GetReadRegions()) {
                child_used_buffers.insert(region.get()->buffer);
              }
              for (const auto &region : other_child->GetWriteRegions()) {
                child_used_buffers.insert(region.get()->buffer);
              }
            }
          }
          ScheduleRecursive(child, child_used_buffers);
        }
        Z3SchedulePythonLoop(ctrl, used_buffers);
      } else {
        ScheduleRecursive(ctrl->child, used_buffers);
        auto old_child = ctrl->child;
        auto seq_node = std::make_shared<SequenceNode>();
        seq_node->children = {old_child};
        ctrl->child = seq_node;
        Z3SchedulePythonLoop(ctrl, used_buffers);
      }
    }
    return;
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<WrapperNode *>(node.get());
    std::vector<std::shared_ptr<IRStructure>> origin_children;
    GatherTaskNodes({wrapper->task, wrapper->child}, origin_children);
    for (auto &child : origin_children) {
      auto child_used_buffers = used_buffers;
      for (auto &other_child : origin_children) {
        if (child.get() != other_child.get()) {
          for (const auto &region : other_child->GetReadRegions()) {
            child_used_buffers.insert(region.get()->buffer);
          }
          for (const auto &region : other_child->GetWriteRegions()) {
            child_used_buffers.insert(region.get()->buffer);
          }
        }
      }
      ScheduleRecursive(child, child_used_buffers);
    }
    auto seq_node = std::make_shared<SequenceNode>();
    seq_node->children = ChildrenScheduleHelper(origin_children);
    int64_t overall_latency = 0;
    for (const auto &child : seq_node->children) {
      overall_latency += child->GetLatency();
    }
    seq_node->SetLatency(overall_latency);
    seq_node->SetII(overall_latency);
    node = seq_node;
    return;
  } else if (node->IsIf()) {
    auto if_node = static_cast<IfNode *>(node.get());
    if (if_node->then_child) {
      ScheduleRecursive(if_node->then_child, used_buffers);
    }
    if (if_node->else_child) {
      ScheduleRecursive(if_node->else_child, used_buffers);
    }
    if_node->SetLatency(
        std::max(if_node->then_child ? if_node->then_child->GetLatency() : 0,
                 if_node->else_child ? if_node->else_child->GetLatency() : 0));
    if_node->SetII(
        std::max(if_node->then_child ? if_node->then_child->GetII() : 0,
                 if_node->else_child ? if_node->else_child->GetII() : 0));
    return;
  }

  LOG(FATAL) << "[ScheduleRecursive] Unknown IRStructure type" << node.get();
}

// --- Naive scheduling implementation ---

std::vector<PrimExpr>
NaiveAssignWarpgroupIds(IRStructure *root, const WarpSpecializeConfig &config,
                        PrimExpr thread_count) {
  if (!root)
    LOG(FATAL) << "Empty root";

  std::vector<TaskNodeWithContext> all_tasks;
  CollectAllTaskNodesWithContext(root, all_tasks);
  if (all_tasks.empty())
    LOG(FATAL) << "No task";

  // Simple producer/consumer assignment:
  // TMA tasks → wg1 (producer), compute tasks → wg0 (consumer)
  for (auto &task_ctx : all_tasks) {
    TaskNode *task = task_ctx.task;
    if (task->ContainsLoopBreak()) {
      task->SetWarpgroupId(kWarpgroupBroadcast);
      continue;
    }
    if (task->UsesTMACore() && !task->UsesTensorCore()) {
      task->SetWarpgroupId(1); // producer
    } else {
      task->SetWarpgroupId(0); // consumer
    }
  }

  // Collect prefix/suffix tasks and reset them to neutral
  std::unordered_set<TaskNode *> prefix_tasks;
  CollectPrefixTasks(root, prefix_tasks);
  for (auto *task : prefix_tasks) {
    task->SetSchedulePhase(SchedulePhase::kPrologue);
    task->SetWarpgroupId(0);
  }

  int n = all_tasks.size();
  TaskUnionFind uf(n);
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      if (UseSameRegisterRegion(all_tasks[i].task, all_tasks[j].task)) {
        uf.unite(i, j);
      }
    }
  }
  std::unordered_set<TaskNode *> suffix_tasks;
  CollectSuffixTasks(root, all_tasks, uf, suffix_tasks);
  for (auto *task : suffix_tasks) {
    task->SetSchedulePhase(SchedulePhase::kEpilogue);
    task->SetWarpgroupId(0);
  }

  // no double_thread in naive mode
  if (config.enable_thread_extend) {
    return {thread_count,
            IntImm(DataType::Int(32), config.producer_thread_count)};
  } else {
    return {IntImm(DataType::Int(32), 32), IntImm(DataType::Int(32), 32),
            thread_count - IntImm(DataType::Int(32), 64)};
  }
}

void ScheduleUnitBuilder::NaiveScheduleLoop(ControlNode *ctrl) {
  if (!ctrl->child)
    return;

  // Flatten children
  std::vector<std::shared_ptr<IRStructure>> flat_children;
  if (!ctrl->child->IsSequence()) {
    GatherTaskNodesSingle(ctrl->child, flat_children);
  } else {
    auto seq_node = static_cast<SequenceNode *>(ctrl->child.get());
    GatherTaskNodes(seq_node->children, flat_children);
  }
  auto seq_node = std::make_shared<SequenceNode>();
  seq_node->children = flat_children;
  ctrl->child = std::move(seq_node);

  auto seq_body = static_cast<SequenceNode *>(ctrl->child.get());

  // Read num_stages from loop annotation
  int num_stages = 1;
  auto num_stages_val = ctrl->control.get()->annotations.Get("num_stages");
  if (num_stages_val.has_value()) {
    num_stages = num_stages_val.value().cast<IntImm>()->value;
  }

  // Assign pipeline stages and start times:
  // - TMA load → stage 0, start_time = 0
  // - Everything else → stage (num_stages - 1), start_time = num_stages
  // - All task latencies set to 0, IIperIter = 1
  std::map<IRStructure *, int> stage_map;
  bool has_promoted = false;
  for (auto &child : seq_body->children) {
    IRStructure *node = child.get();
    bool is_tma_load =
        node->UsesTMACore() && !node->UsesTensorCore() && !node->UsesCUDACore();
    if (is_tma_load && node->IsTask()) {
      is_tma_load = static_cast<TaskNode *>(node)->HasTMALoad();
    }
    int stage = !is_tma_load ? 0 : (num_stages);
    stage_map[node] = stage;
    if (stage != num_stages) {
      has_promoted = true;
    }
    node->SetStartTime(is_tma_load ? 0 : num_stages);
    node->SetLatency(0);
    node->SetII(0);
  }

  ctrl->SetIIperIter(1);

  int n = static_cast<int>(seq_body->children.size());
  auto IsVarDecl = [](IRStructure *node) -> bool {
    if (!node || !node->IsTask())
      return false;
    auto task = static_cast<TaskNode *>(node);
    return task->stmts.size() == 1 &&
           task->stmts[0].as<LetStmtNode>() != nullptr;
  };
  auto SolveConflictVar = [&]() -> bool {
    for (int i = 0; i < n; ++i) {
      if (!IsVarDecl(seq_body->children[i].get()))
        continue;
      for (int j = 0; j < n; ++j) {
        if (i == j)
          continue;
        auto node_i = seq_body->children[i].get();
        auto node_j = seq_body->children[j].get();
        if (!HasDependency(node_i, node_j))
          continue;
        if (stage_map[node_j] == stage_map[node_i])
          continue;

        int rem_stage_j = stage_map[node_j];
        auto node_i_task = static_cast<TaskNode *>(node_i);
        auto node_i_let_stmt = node_i_task->stmts[0].as<LetStmtNode>();

        auto cloned_let_stmt =
            LetStmt(node_i_let_stmt->var.copy_with_suffix(""),
                    node_i_let_stmt->value, Evaluate(0));
        auto cloned_task = std::make_shared<TaskNode>();
        cloned_task->stmts.push_back(cloned_let_stmt);
        cloned_task->SetReadRegions(node_i_task->GetReadRegions());
        cloned_task->SetWriteRegions(node_i_task->GetWriteRegions());
        cloned_task->SetReadVars(node_i_task->GetReadVars());
        {
          auto write_vars = node_i_task->GetWriteVars();
          for (auto &v : write_vars) {
            if (v.same_as(node_i_let_stmt->var)) {
              v = cloned_let_stmt->var;
            }
          }
          cloned_task->SetWriteVars(write_vars);
        }
        cloned_task->SetLatency(node_i_task->GetLatency());
        cloned_task->SetII(node_i_task->GetII());
        cloned_task->SetUsesCUDACore(node_i_task->UsesCUDACore());
        cloned_task->SetUsesTMACore(node_i_task->UsesTMACore());
        cloned_task->SetUsesTensorCore(node_i_task->UsesTensorCore());
        stage_map[cloned_task.get()] = rem_stage_j;

        for (int k = j; k < n; ++k) {
          auto node_k = seq_body->children[k].get();
          if (rem_stage_j != stage_map[node_k])
            continue;
          if (HasDependency(node_i, node_k)) {
            node_k->SubstituteVar(node_i_let_stmt->var, cloned_let_stmt->var);
            stage_map[node_k] = rem_stage_j;
          }
        }

        seq_body->children.insert(seq_body->children.begin() + j,
                                  std::move(cloned_task));
        n += 1;
        return true;
      }
    }
    return false;
  };
  int conflict_count = 0;
  while (SolveConflictVar() && ++conflict_count < 100)
    ;

  // Estimate overall latency
  int64_t tripcount = 100;
  const ForNode *for_node = ctrl->control.get();
  PrimExpr loop_extent = for_node->extent;
  PrimExpr loop_step = for_node->step.has_value()
                           ? for_node->step.value()
                           : IntImm(DataType::Int(32), 1);
  if (const auto *extent_int = loop_extent.as<IntImmNode>()) {
    if (const auto *step_int = loop_step.as<IntImmNode>()) {
      int64_t extent = extent_int->value;
      int64_t step = step_int->value;
      if (step > 0) {
        tripcount = (extent + step - 1) / step;
      }
    }
  }
  ctrl->SetII(tripcount);
  ctrl->SetLatency(tripcount);

  // Wrap in ScheduleUnits with assigned stages
  for (auto &node : seq_body->children) {
    auto unit = std::make_shared<ScheduleUnit>();
    unit->stage = stage_map[node.get()];
    unit->child = std::move(node);
    node = std::move(unit);
  }

  if (has_promoted) {
    ctrl->SetPromote(true);
  }
}

void ScheduleUnitBuilder::NaiveScheduleRecursive(
    std::shared_ptr<IRStructure> &node) {
  if (!node)
    return;

  // Helper to wrap children in ScheduleUnits preserving order
  auto WrapInScheduleUnits =
      [](std::vector<std::shared_ptr<IRStructure>> &children) {
        for (auto &child : children) {
          auto unit = std::make_shared<ScheduleUnit>();
          unit->stage = -1;
          unit->child = std::move(child);
          child = std::move(unit);
        }
      };

  if (node->IsTask()) {
    return;
  } else if (node->IsSequence()) {
    auto seq = static_cast<SequenceNode *>(node.get());
    std::vector<std::shared_ptr<IRStructure>> origin_children;
    GatherTaskNodes(seq->children, origin_children);
    for (auto &child : origin_children) {
      NaiveScheduleRecursive(child);
    }
    WrapInScheduleUnits(origin_children);
    seq->children = origin_children;
  } else if (node->IsControl()) {
    auto ctrl = static_cast<ControlNode *>(node.get());
    if (ctrl->child) {
      if (ctrl->child->IsSequence() || ctrl->child->IsWrapper()) {
        std::vector<std::shared_ptr<IRStructure>> origin_children;
        if (ctrl->child->IsSequence()) {
          auto seq_body = static_cast<SequenceNode *>(ctrl->child.get());
          GatherTaskNodes(seq_body->children, origin_children);
        } else {
          auto wrapper = static_cast<WrapperNode *>(ctrl->child.get());
          GatherTaskNodes({wrapper->task, wrapper->child}, origin_children);
        }
        for (auto &child : origin_children) {
          NaiveScheduleRecursive(child);
        }
        NaiveScheduleLoop(ctrl);
      } else {
        NaiveScheduleRecursive(ctrl->child);
      }
    }
  } else if (node->IsWrapper()) {
    auto wrapper = static_cast<WrapperNode *>(node.get());
    std::vector<std::shared_ptr<IRStructure>> origin_children;
    GatherTaskNodes({wrapper->task, wrapper->child}, origin_children);
    for (auto &child : origin_children) {
      NaiveScheduleRecursive(child);
    }
    auto seq_node = std::make_shared<SequenceNode>();
    WrapInScheduleUnits(origin_children);
    seq_node->children = origin_children;
    node = seq_node;
  } else if (node->IsIf()) {
    // IfNode: recursively schedule both branches internally
    auto if_node = static_cast<IfNode *>(node.get());
    if (if_node->then_child) {
      NaiveScheduleRecursive(if_node->then_child);
    }
    if (if_node->else_child) {
      NaiveScheduleRecursive(if_node->else_child);
    }
  } else {
    LOG(FATAL) << "[NaiveScheduleRecursive] Unknown IRStructure type";
  }
}

std::vector<PrimExpr>
ScheduleUnitBuilder::NaiveBuild(std::shared_ptr<IRStructure> &root) {
  NaiveScheduleRecursive(root);
  auto result =
      NaiveAssignWarpgroupIds(root.get(), config_, thread_var_->dom->extent);
  PropagateBroadcastWarpgroupId(root.get());
  return result;
}

} // namespace tl
} // namespace tvm
