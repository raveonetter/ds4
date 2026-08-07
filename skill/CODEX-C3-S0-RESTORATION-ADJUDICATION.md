# Codex C3 S0 Restoration Adjudication

This is a source-only adjudication. It does not implement C3, add probe hooks,
start C2b, remove replay, or authorize a production optimization. No conclusion
below is `PROVEN BY TEST`.

| Audit item | Value |
|---|---|
| Repository | `sijiaguo777/ds4` |
| Production-source baseline | `b0309611041655f4e45671cfd9c9886aff161406` |
| Documentation branch base | validated C2a candidate `34f858fce638dff779c883ada3354339cd07a95d` |
| Authoritative Pass B | ordinary sequential target decode through `metal_graph_eval_token_raw_swa(...)` |
| Primary experiment scope | single-device Metal, matching the validated M4 Max C2a context |
| Upstream check | `antirez/ds4` `main` still resolves to `b0309611041655f4e45671cfd9c9886aff161406` |

The source baseline and the C2a candidate have the same `ds4.c`. C2a adds the
copy primitive and tests but does not change the Pass-A or Pass-B call graph.

## A. Authoritative S0 definition

S0 is the effective target execution state immediately before the caller
temporarily appends the proposed suffix and invokes the generic verifier. It is
defined by values that ordinary sequential Pass B can read or branch on before
Pass B overwrites or reinitializes them.

For the target model this includes:

- every live raw-KV physical row that Pass A can overwrite and Pass B can read;
- attention-compressor frontier tensors and `layer_n_comp[]`;
- ratio-4 indexer frontier tensors and `layer_n_index_comp[]`;
- the graph metadata that selects an execution tier or makes cache rows visible;
- the logical checkpoint length used as the explicit decode position;
- canonical host logits for the state preceding the first forced token.

It does not require byte restoration of work buffers that Pass B cannot read,
or append-cache rows hidden by restored counts and overwritten before their
first subsequent read.

The graph has no implicit target position counter. Pass B receives `pos`
explicitly from `s->checkpoint.len`; raw addressing is derived from `pos`,
`raw_cap`, and `raw_window`. **PROVEN BY SOURCE** (`metal_graph_eval_token_raw_swa`,
`metal_graph_encode_token_raw_swa`, `metal_graph_raw_span_for_batch`).

## B. Pass-A mutation inventory

The actual caller path is:

```text
ds4_session_eval_dspark_speculative_argmax
  -> append drafts to s->checkpoint
  -> metal_graph_verify_suffix_tops
     -> metal_graph_verify_suffix_tops_impl
        -> upload draft ids and embeddings into batch work tensors
        -> for each layer: metal_graph_encode_layer_batch
           -> metal_graph_encode_layer_attention_batch
           -> metal_graph_encode_layer_ffn_batch
        -> metal_graph_encode_output_head_batch
        -> write/read verifier top rows
```

The mutation inventory below follows that call graph, rather than inferring
semantics from names.

| Object | Allocation / owner | Exact Pass-A writer | First relevant post-A reader | Persistent or scratch | Snapshot / restore coverage | Finding |
|---|---|---|---|---|---|---|
| `layer_raw_cache[il]` | per-layer tensor owned by `ds4_gpu_graph`; `raw_cap * DS4_N_HEAD_DIM` F32 elements | `ds4_gpu_store_raw_kv_batch_tensor` from the nonzero batch-attention branches | Pass-B attention kernels through `metal_graph_encode_decode_layer` | persistent target KV | neither function | destination rows that alias still-live S0 history are `MUST RESTORE`; conservative C3 snapshots every Pass-A destination row. **PROVEN BY SOURCE** |
| `layer_attn_state_kv/score[il]` | per compressed layer, graph-owned full frontier tensors | generic batch compressor prefill/replay/update kernels | Pass-B compressor update for its first token | persistent target frontier | full tensor copy both ways | `MUST RESTORE`. **PROVEN BY SOURCE** |
| `layer_n_comp[il]` | scalar array in `ds4_gpu_graph` | generic batch compressor branches | capacity checks, append-row selection, and attention visibility in Pass B | persistent target frontier | saved and restored | `MUST RESTORE`. **PROVEN BY SOURCE** |
| `layer_attn_comp_cache[il]` | per compressed layer, graph-owned append cache | generic batch compressor output/commit | Pass-B attention, bounded by restored `layer_n_comp` | persistent target cache | contents not copied | stale appended rows are hidden, then overwritten before becoming visible. `INVISIBLE AFTER RESTORE`. **PROVEN BY SOURCE** |
| `layer_index_state_kv/score[il]` | per ratio-4 layer, graph-owned full frontier tensors | generic batch indexer compressor prefill/replay/update kernels | Pass-B ratio-4 compressor update | persistent target frontier | full tensor copy both ways | `MUST RESTORE`. **PROVEN BY SOURCE** |
| `layer_n_index_comp[il]` | scalar array in `ds4_gpu_graph` | generic batch ratio-4 branches | Pass-B append-row selection and indexer score/top-k bounds | persistent target frontier | saved and restored | `MUST RESTORE`. **PROVEN BY SOURCE** |
| `layer_index_comp_cache[il]` | per ratio-4 layer, graph-owned append cache | generic batch indexer compressor | Pass-B indexer score kernel, bounded by restored counter | persistent target cache | contents not copied | stale appended rows are hidden, then overwritten before becoming visible. `INVISIBLE AFTER RESTORE`. **PROVEN BY SOURCE** |
| decode `cur_hc_by_tier[]` / `after_ffn_hc_by_tier[]` contents and pointer identities | graph-owned decode work tensors | no writer in the generic batch verifier | embedding and decode layers in Pass B | decode work state | not handled | contents are `OVERWRITTEN BEFORE READ`; pointer identity is unchanged and outside the restoration set. **PROVEN BY SOURCE** |
| batch HC/Q/KV/FFN work tensors | graph-owned batch tensors | upload, layer attention, layer FFN, and batch head kernels | no target Pass-B reader; Pass B uses decode tensors | batch scratch | not handled | `SCRATCH / NOT PART OF S0`. **PROVEN BY SOURCE** |
| `batch_cur_hc_by_tier[]` / `batch_next_hc_by_tier[]` pointer identities | graph-owned batch pointer pairs | swapped after each generic batch layer | only a later batch operation | batch scratch metadata | not handled | Pass B does not use these accessors. `SCRATCH / NOT PART OF S0`. **PROVEN BY SOURCE** |
| shared selection/index score work buffers | graph-owned work tensors such as `comp_selected` and `indexer_scores` | generic verifier attention and verifier argmax | Pass B kernels overwrite scores/selections before consuming them | scratch | not handled | `OVERWRITTEN BEFORE READ`. **PROVEN BY SOURCE** |
| `active_tier` and current GPU device | scalar graph metadata plus backend device selection | `metal_graph_set_active_tier_batch` while layers cross tiers | `metal_graph_set_active_tier_decode` at Pass-B embedding-tier entry | execution metadata | not handled | single-device Metal leaves it at 0; a placement/multi-tier C3 must save and restore it with device selection. **PROVEN BY SOURCE** |
| `spec_capture_prefixes` | graph scalar | verifier saves, changes, then restores it | later verifier/decode control flow | transient control metadata | self-restored by verifier | no C3 action. **PROVEN BY SOURCE** |
| `tp_batch_rows` | graph scalar | verifier sets it for the batch and resets to 0 | later TP dispatch | transient control metadata | self-reset by verifier | no C3 action for the single-device experiment. **PROVEN BY SOURCE** |
| `spec_prefix1_*` tensors and prefix counters | graph-owned verifier snapshots | per-token generic compressor capture | no Pass-B target reader | verifier scratch | not handled by frontier restore | `SCRATCH / NOT PART OF S0`. **PROVEN BY SOURCE** |
| `dspark_target_hidden` | graph-owned DSpark capture tensor | `metal_graph_dspark_capture_verified_suffix_layer` copies each selected layer's last verified row | DSpark proposer, not target forward; Pass-B capture overwrites every selected slot first | proposer input / capture scratch | not handled | `OVERWRITTEN BEFORE READ` for an ordinary Pass B followed by normal checkpoint-note flow. **PROVEN BY SOURCE** |
| `dspark_target_hidden_batch` | graph-owned DSpark batch capture tensor | suffix-begin copies S0 row 0; suffix-layer writes rows 1..N | batch proposer only when batch-capture metadata is valid | proposer/capture scratch | not handled | caller invalidates batch-capture metadata, making stale rows unreachable. `INVISIBLE AFTER RESTORE`. **PROVEN BY SOURCE** |
| DSpark capture masks, ranges, validity, checkpoint length | graph scalar metadata | verified-suffix capture and caller invalidation | proposer readiness checks | proposer/capture metadata | not handled | current caller deliberately invalidates; Pass-B target math does not branch on it. Re-entry to scheduling must use Pass-B recapture/note or remain isolated. Target-state fact is source-proven; scheduling isolation is **INFERRED** |
| `dspark_raw_cache[]` and DSpark cache window contents | support-model tensors owned by graph | no writer in target verifier call graph | future proposer | support-model persistent state | cache window scalars are saved/restored; contents are not | no Pass-A content mutation. `SCRATCH / NOT PART OF S0` for target S0. **PROVEN BY SOURCE** |
| MTP `mtp_raw_cache` and MTP work tensors | optional graph-owned support-model state | no writer in target verifier call graph | MTP proposer only | support-model persistent/scratch state | contents not handled | no Pass-A mutation. `SCRATCH / NOT PART OF S0` for target S0. **PROVEN BY SOURCE** |
| `mtp_n_raw` | graph scalar | no writer in target verifier | MTP proposer | support-model frontier | saved/restored | redundant for this Pass A but exactly preserved. **PROVEN BY SOURCE** |
| `s->checkpoint.len` | logical length in session-owned `token_vec` | caller appends drafts | caller supplies Pass-B `pos` from the length | target execution metadata | not handled by frontier functions | reset to `start` before Pass B: `MUST RESTORE`. **PROVEN BY SOURCE** |
| checkpoint storage beyond `len`, pointer, capacity | session-owned `token_vec` allocation | `token_vec_push`, possibly including reallocation | only indexed below logical `len`; later push overwrites next slot | host container storage | not handled | stale token values are `INVISIBLE AFTER RESTORE`; allocation identity/capacity is not model state. **PROVEN BY SOURCE** |
| `checkpoint_valid` | session scalar | no Pass-A writer on the successful verifier path | later session control flow | session metadata | not handled | remains at S0 value; no C3 action. **PROVEN BY SOURCE** |
| `g->spec_logits` | graph-owned verifier output tensor | batch output head | verifier top/readback only | verifier scratch | not handled | `SCRATCH / NOT PART OF S0`. **PROVEN BY SOURCE** |
| `s->spec_row_logits` | session-owned replay output buffer | no Pass-A writer; the caller passes `NULL` for verifier row-logit readback | ordinary Pass B writes its output logits here | replay scratch | not handled | `OVERWRITTEN BEFORE READ`. **PROVEN BY SOURCE** |
| row tops and verifier selection buffers | caller stack / graph scratch | verifier argmax and readback | acceptance logic only | verifier scratch | not handled | `SCRATCH / NOT PART OF S0`. **PROVEN BY SOURCE** |
| `s->logits` | canonical host session logits | no Pass-A writer | first-draft comparison and ordinary scheduling | canonical session state | not handled | preserved without action. **PROVEN BY SOURCE** |
| decode GPU logits | graph output-head tensor | not written by single-device Metal batch head, which targets `spec_logits` | Pass-B output head writes before host read | target output workspace | not handled | preserved on single-device Metal; output-TP scratch shards, if used, are overwritten by Pass B before read. **PROVEN BY SOURCE** |

## C. `spec_frontier_snapshot/restore` exact coverage

`spec_frontier_snapshot()` performs exactly these actions:

1. validates the current DSpark support-cache window;
2. saves `mtp_n_raw`;
3. saves `dspark_cache_start`, `dspark_cache_token_start`, and
   `dspark_cache_len`;
4. for every layer, saves `layer_n_comp[il]` and
   `layer_n_index_comp[il]`;
5. for every compressed layer, copies all bytes of
   `layer_attn_state_kv[il]` and `layer_attn_state_score[il]` into
   `spec_attn_state_*[il]`;
6. for every ratio-4 layer, copies all bytes of
   `layer_index_state_kv[il]` and `layer_index_state_score[il]` into
   `spec_index_state_*[il]`.

`spec_frontier_restore()` performs exactly these actions:

1. validates the saved DSpark cache tuple;
2. restores `mtp_n_raw`;
3. calls `metal_graph_dspark_cache_set_window()` with the saved token start
   and length; this recomputes `dspark_cache_start`, and the preceding
   validation proves it equals the saved physical start;
4. restores both compressed-cache counters for every layer;
5. copies all saved attention frontier bytes back for every compressed layer;
6. copies all saved indexer frontier bytes back for every ratio-4 layer.

| State | Snapshot function handles it? | Restore function handles it? | Additional C3 action needed? |
|---|---:|---:|---|
| `layer_attn_state_kv[]` | yes, full tensor | yes, full tensor | no |
| `layer_attn_state_score[]` | yes, full tensor | yes, full tensor | no |
| `layer_n_comp[]` | yes | yes | no |
| `layer_attn_comp_cache[]` | no | no | no; restored count hides stale append rows and sequential reuse overwrites first |
| `layer_index_state_kv[]` | yes on ratio-4 layers | yes on ratio-4 layers | no |
| `layer_index_state_score[]` | yes on ratio-4 layers | yes on ratio-4 layers | no |
| `layer_n_index_comp[]` | yes | yes | no |
| `layer_index_comp_cache[]` | no | no | no; same proven visibility order, independently audited |
| `layer_raw_cache[]` | no | no | yes, snapshot before Pass A and restore before Pass B |
| `raw_cap`, `raw_window` | no | no | no; immutable during both passes |
| raw-cache base/current-position scalar | none exists | none exists | no; addressing derives from explicit `pos` |
| `mtp_n_raw` | yes | yes | no |
| `mtp_raw_cache` | no | no | no; Pass A does not write it |
| `dspark_cache_start` | yes | recomputed from saved token start | no |
| `dspark_cache_token_start` | yes | yes | no |
| `dspark_cache_len` | yes | yes | no |
| `dspark_raw_cache[]` contents | no | no | no; Pass A does not write them |
| decode HC pointer identity | no | no | no; Pass A does not change it |
| batch HC pointer identity | no | no | no for target Pass B |
| `active_tier` / current device | no | no | none on single-device Metal; save/restore in a multi-tier extension |
| DSpark hidden capture and validity metadata | no | no | no for isolated target Pass B; invalidate and regenerate before scheduler reuse |
| `s->checkpoint.len` | no | no | yes, set to `start` |
| `checkpoint_valid` | no | no | no; Pass A preserves it |
| `s->logits` | no | no | no; Pass A preserves it |
| `spec_logits` and verifier row buffers | no | no | no; scratch |

Every table entry above is **PROVEN BY SOURCE** except the stated scheduling
isolation condition, which is **INFERRED** from the intended diagnostic control
flow.

## D. Raw-KV ring analysis

### Physical writer formula

For every target layer, the generic verifier stores each suffix row with:

```text
physical_row(t) = (start + t) % raw_cap
t = 0 .. n_tokens - 1
row_bytes = DS4_N_HEAD_DIM * sizeof(float)
```

The C caller passes `pos0=start`; the Metal implementation constructs exactly
that row list before `set_rows`, and the CUDA kernel uses the same formula.
**PROVEN BY SOURCE** (`ds4_gpu_store_raw_kv_batch_tensor`).

The generic verifier writes all suffix raw rows before the mixed batch
attention consumes them. This is materially different from Pass B, which
writes one row and then runs that token's attention. **PROVEN BY SOURCE**.

### `raw_cap` versus `raw_window`

`raw_window` is the logical SWA window. `raw_cap` is the physical ring. The
normal planner requests `raw_window + prefill_cap`, aligns to 256, and caps at
the context/8192 limit. An environment override may reduce the physical ring
to exactly `raw_window`. Therefore physical headroom is common but not an
invariant. **PROVEN BY SOURCE** (`metal_graph_raw_cap_for_context`).

For Pass B token at logical position `p`, the visible raw span is computed by
`metal_graph_raw_span_for_batch(g, p, 1)`, and its first physical row is:

```text
first_logical = p + 1 - n_raw
raw_start = first_logical % raw_cap
```

If `raw_cap` has enough headroom, speculative future rows lie outside the
logical raw span until Pass B overwrites each row itself. If `raw_cap` is
strict or nearly strict, a future speculative row can alias an S0 historical
row that an early Pass-B token still needs. For example, with
`raw_cap == raw_window == W`, Pass A row `t=1` overwrites the physical row for
logical position `start-W+1`, which token `start` still reads. **PROVEN BY
SOURCE**.

### Rows that require preservation

The mathematically minimal set is the intersection of:

```text
Pass-A destinations:
  D = { (start+t) % raw_cap | 0 <= t < n_tokens }

S0 historical rows read by Pass B before their canonical overwrite:
  H = physical rows for logical positions
      max(0, start-(raw_window-1)) .. start-1,
      restricted by metal_graph_raw_span_for_batch at each Pass-B position
```

Rows in `D ∩ H` are `MUST RESTORE`. Proving and encoding only that subset adds
configuration-sensitive logic with no diagnostic benefit. The minimal robust
C3 design therefore snapshots and restores all distinct rows in `D` for every
target layer. This is conservative in bytes and exact in semantics. **PROVEN
BY SOURCE** for necessity of `D ∩ H`; choosing all of `D` is a design
inference.

The draft block is at most 16 rows and the physical ring is at least the raw
window, so these destination rows are distinct in the experiment. Let
`r = start % raw_cap`:

- if `r + n_tokens <= raw_cap`, copy one contiguous segment `[r, r+n_tokens)`;
- otherwise copy `[r, raw_cap)` and `[0, (r+n_tokens) % raw_cap)`.

Thus restoration may require two wrapped segments. **PROVEN BY SOURCE**.

`spec_frontier_restore()` alone is not sufficient to restore raw target KV in
all supported configurations. It is sufficient only under a separately proven
no-alias condition for the exact `start`, block length, `raw_cap`, and
`raw_window`. C3 should not depend on that condition. **PROVEN BY SOURCE**.

## E. Attention compressor state

Pass A projects all batch rows, updates `layer_attn_state_kv/score`, appends at
each emission boundary, and advances `layer_n_comp`. The frontier tensors and
counter are fully restored.

Suppose the S0 compressed count is `c`. After restore:

1. all attention readers receive a count no greater than `c`, so physical row
   `c` and later rows cannot be addressed;
2. non-emitting sequential tokens update only the restored frontier and do not
   make an appended row visible;
3. at the first emitting token, the sequential compressor selects append row
   `c`, computes the canonical row, quantizes/commits it, and only then
   increments `layer_n_comp`;
4. the following attention sees the incremented count only after row `c` has
   canonical contents;
5. every later emission repeats the same ordering for `c+1`, `c+2`, and so on.

This proof covers multiple emissions and blocks beginning midway through a
ratio cycle. It does not rely on the phrase "append-only" alone. Stale
compressed-cache contents are `INVISIBLE AFTER RESTORE`, and no byte copy of
`layer_attn_comp_cache` is required. **PROVEN BY SOURCE**.

On Metal, the attention compressed cache may be F16 behind an F32 staging
tensor. The sequential path still commits the staging row to the persistent
append row before incrementing visibility. The staging tensor is scratch and
is overwritten by the next compressor update. **PROVEN BY SOURCE**.

## F. Ratio-4 indexer state

The ratio-4 argument is independently established:

- Pass A writes `layer_index_state_kv/score`, advances
  `layer_n_index_comp`, and appends to `layer_index_comp_cache`;
- snapshot/restore copies the complete ratio-4 frontier tensors and restores
  the index count;
- indexer score/top-k receives `layer_n_index_comp` as its row bound;
- after restore, stale row `c` is outside that bound;
- at an emission, sequential `ds4_gpu_compressor_update_tensor` writes row
  `c`, QAT transforms that row, and only then increments the count;
- indexer scoring occurs afterward and may now read the canonical row.

The same visibility result therefore holds for the indexer cache, but through
its own writer/counter/scorer sequence. Multiple emits, a block starting
mid-cycle, and an already-active sparse indexer do not change the ordering.
`layer_index_comp_cache` is `INVISIBLE AFTER RESTORE`; its frontier and counter
are `MUST RESTORE` and are already covered. **PROVEN BY SOURCE**.

## G. HC / batch / decode scratch

### Tensor contents

Pass A overwrites the batch HC, Q/KV, attention, FFN, router, selection, and
output-head work tensors. Ordinary Pass B enters through the decode accessors,
embeds the forced token into decode `cur_hc`, and then overwrites each decode
stage before consuming it. Batch work contents cannot feed the target forward
path. **PROVEN BY SOURCE**.

### Pointer identity

`metal_graph_encode_layer_batch` swaps only
`batch_cur_hc_by_tier[]`/`batch_next_hc_by_tier[]`. It does not swap decode
`cur_hc_by_tier[]`/`after_ffn_hc_by_tier[]`. The latter pointer identity is
therefore still exactly S0. Batch pointer orientation is not selected by Pass
B. **PROVEN BY SOURCE**.

### Active view / tier metadata

`active_tier` is shared between batch and decode dispatch. On single-device
Metal, `placement == NULL`, all tier setters are no-ops, and `active_tier`
remains 0. No restoration is needed for the M4 experiment.

With a non-null placement, Pass A changes `active_tier`. Pass B branches on the
old value inside `metal_graph_set_active_tier_decode` before setting the
embedding tier. Although the copied decode HC is subsequently overwritten by
embedding, the task's strict future-visible criterion includes branching on
modified metadata. A multi-tier C3 must save the S0 tier and restore it with
`metal_graph_set_active_tier_no_copy`, which also restores backend current
device selection; direct scalar assignment is insufficient. **PROVEN BY
SOURCE**.

## H. Support-model / MTP state

The generic verifier call receives the target model and target weights. Its
call graph contains no DSpark proposal stage and no MTP forward stage.
Consequently it does not write:

- `dspark_raw_cache[]`;
- MTP `mtp_raw_cache`;
- MTP hidden work tensors;
- `mtp_n_raw`.

The frontier functions still preserve `mtp_n_raw` and the DSpark support-cache
window metadata. This is conservative. DSpark raw-cache contents remain
unchanged. Pass B target decode does not depend on either support cache for its
forward math. **PROVEN BY SOURCE**.

`dspark_target_hidden` and `_batch` are distinct: they are captures produced by
the target forward path for later proposer consumption. They are not target KV
or recurrent target hidden state. Pass B target layers never read them.
**PROVEN BY SOURCE**.

## I. Checkpoint, logits, metadata

### Checkpoint vector

The caller appends the exact drafts so the verifier can read
`prompt[start..start+n_tokens)`. Resetting `s->checkpoint.len = start` restores
the only logical boundary subsequently used for token lookup and Pass-B
position. Values beyond `len` cannot be reached by the token-vector contract,
and the next `token_vec_push` overwrites the next logical slot. Reallocation or
capacity growth is not inference state. Therefore resetting `len` is
sufficient to make stale token-vector contents invisible. **PROVEN BY SOURCE**.

### Logits

The generic verifier's batch output head targets `g->spec_logits`. It reads
row tops into verifier buffers. It does not copy any verifier row into
`s->logits`; canonical session logits remain the S0 logits used to compare the
first draft. Pass B computes new logits through the ordinary output head.
No session-logit restoration is needed. **PROVEN BY SOURCE**.

### Cache/window metadata

- `raw_cap` and `raw_window` are allocation/configuration fields and are not
  changed by Pass A;
- there is no raw-cache base or current-position scalar;
- compressed visibility is entirely controlled by the restored counters;
- DSpark support-cache start/token-start/length are restored;
- `spec_capture_prefixes` is restored by the verifier;
- `tp_batch_rows` is reset by the verifier;
- sparse-attention threshold selection is configuration/static state, not
  mutated by Pass A.

**PROVEN BY SOURCE**.

### Scheduler, statistics, and host metadata

Verifier timing/statistics and selected-profile counters do not feed target
math. `dspark_stats` fields are pure statistics. The DSpark scheduler state is
updated outside the generic verifier pass in the session wrapper; it is not
part of target S0.

The diagnostic must terminate or isolate before stale proposal/capture
metadata can influence a later scheduling decision, or must complete the
ordinary recapture/checkpoint-note sequence before returning. That isolation
requirement is **INFERRED**; it is not a claim that scheduler state is target
model state.

## J. Mandatory restoration table

| Object | Pass A writes? | Pass B can observe before overwrite? | Current restore coverage | Classification | Evidence |
|---|---:|---:|---|---|---|
| aliased live rows in `layer_raw_cache[]` | yes | yes, under strict/nearly strict ring headroom | none | MUST RESTORE | PROVEN BY SOURCE |
| remaining Pass-A raw destination rows | yes | no; Pass B overwrites its own row first | none | OVERWRITTEN BEFORE READ | PROVEN BY SOURCE |
| `layer_attn_state_kv/score[]` | yes | yes | full tensor | MUST RESTORE | PROVEN BY SOURCE |
| `layer_n_comp[]` | yes | yes | scalar | MUST RESTORE | PROVEN BY SOURCE |
| appended `layer_attn_comp_cache[]` rows | yes, at emits | no while stale; count hides them and first reuse overwrites | counter only | INVISIBLE AFTER RESTORE | PROVEN BY SOURCE |
| `layer_index_state_kv/score[]` | yes on ratio-4 | yes | full tensor | MUST RESTORE | PROVEN BY SOURCE |
| `layer_n_index_comp[]` | yes on ratio-4 | yes | scalar | MUST RESTORE | PROVEN BY SOURCE |
| appended `layer_index_comp_cache[]` rows | yes, at emits | no while stale; count hides them and first reuse overwrites/QATs | counter only | INVISIBLE AFTER RESTORE | PROVEN BY SOURCE |
| decode `cur_hc` / `after_ffn_hc` contents | no | embedding/stages overwrite before target read | none | OVERWRITTEN BEFORE READ | PROVEN BY SOURCE |
| decode HC pointer identity | no | yes, but it retains S0 identity | none | SCRATCH / NOT PART OF S0 | PROVEN BY SOURCE |
| batch HC/Q/KV/FFN tensor contents | yes | no target Pass-B path reaches them | none | SCRATCH / NOT PART OF S0 | PROVEN BY SOURCE |
| batch HC pointer identity | yes | no target Pass-B path reaches it | none | SCRATCH / NOT PART OF S0 | PROVEN BY SOURCE |
| shared score/selection scratch | yes | no; producer overwrites before consumer | none | OVERWRITTEN BEFORE READ | PROVEN BY SOURCE |
| `active_tier` on single-device Metal | no | constant 0 | none | SCRATCH / NOT PART OF S0 | PROVEN BY SOURCE |
| `active_tier` with placement | yes | yes, tier setter branches on it | none | MUST RESTORE | PROVEN BY SOURCE |
| `spec_capture_prefixes` | temporarily | no; verifier restores it | self-restored | INVISIBLE AFTER RESTORE | PROVEN BY SOURCE |
| prefix-capture tensors/counters | yes | no | none | SCRATCH / NOT PART OF S0 | PROVEN BY SOURCE |
| `dspark_target_hidden` | yes | target decode does not read; recapture overwrites before proposer | none | OVERWRITTEN BEFORE READ | PROVEN BY SOURCE |
| `dspark_target_hidden_batch` stale rows | yes | capture-valid metadata is cleared | none | INVISIBLE AFTER RESTORE | PROVEN BY SOURCE |
| DSpark capture metadata | yes / invalidated | target forward does not read it | none | SCRATCH / NOT PART OF S0 | PROVEN BY SOURCE |
| DSpark support raw cache | no | target Pass B does not read it | none | SCRATCH / NOT PART OF S0 | PROVEN BY SOURCE |
| MTP cache/work tensors | no | target Pass B does not read them | none | SCRATCH / NOT PART OF S0 | PROVEN BY SOURCE |
| `mtp_n_raw` | no | target Pass B does not read it | scalar restored | SCRATCH / NOT PART OF S0 | PROVEN BY SOURCE |
| immutable raw-cap/window metadata | no | yes, unchanged | none | SCRATCH / NOT PART OF S0 | PROVEN BY SOURCE |
| compressed counts | yes | yes | scalar restored | MUST RESTORE | PROVEN BY SOURCE |
| `s->checkpoint.len` | yes, via pushes | yes, supplies Pass-B position | caller reset | MUST RESTORE | PROVEN BY SOURCE |
| checkpoint storage beyond `len` | yes | no | logical length only | INVISIBLE AFTER RESTORE | PROVEN BY SOURCE |
| `checkpoint_valid` | no | yes, unchanged | none | SCRATCH / NOT PART OF S0 | PROVEN BY SOURCE |
| `s->logits` | no | yes, unchanged | none | SCRATCH / NOT PART OF S0 | PROVEN BY SOURCE |
| `spec_logits` and row tops | yes | no target Pass-B input consumes them | none | SCRATCH / NOT PART OF S0 | PROVEN BY SOURCE |
| `s->spec_row_logits` | no | ordinary Pass B overwrites it | none | OVERWRITTEN BEFORE READ | PROVEN BY SOURCE |
| pure stats/profile counters | yes | no target-math read | none | SCRATCH / NOT PART OF S0 | PROVEN BY SOURCE |

Rows of one object are split only where the classification differs physically:
aliased live raw rows are `MUST RESTORE`; other speculative raw destinations
are overwritten before read. C3 should conservatively copy the union.

For fields that Pass A does not modify, `SCRATCH / NOT PART OF S0` in this
table means "not part of the C3 restoration set"; it does not imply that an
unchanged canonical field such as `s->logits` has no semantic value.

## K. Minimal restoration sequence

The minimal robust sequence for the single-device Metal experiment is:

```text
Before Pass A:
1. Record start and snapshot, for every target layer, all physical raw rows
   (start+t) % raw_cap for t=0..n_tokens-1 into dedicated non-aliasing storage.
2. spec_frontier_snapshot(...).

Pass A:
3. Append the forced draft vector and run the generic verifier.

Before Pass B:
4. Set checkpoint.len = start; invalidate verifier-produced DSpark capture;
   spec_frontier_restore(...).
5. Restore the saved target raw-KV rows, using one segment or two on wrap.
6. Run ordinary sequential forced-token decode through
   metal_graph_eval_token_raw_swa(...), pushing each token only after its eval.
```

For a multi-tier extension, step 1 also saves the S0 `active_tier`; step 5
restores it and backend device selection with
`metal_graph_set_active_tier_no_copy` before Pass B.

Both snapshots must finish before Pass A, and both restores must finish before
Pass B. Those boundary orderings are required. Within each side, raw-KV and
frontier copies have no source-level data dependency, but the shown order is
fail-closed and matches the ownership boundary: capture raw bytes first, then
capture the structured frontier; restore the validated structured frontier
first, then restore raw bytes. If frontier restoration fails, Pass B must not
run. **PROVEN BY SOURCE** for the pass boundaries; the relative ordering of
the two independent copy groups is **INFERRED** as the safest implementation
discipline.

## L. Adversarial edge cases

### Raw ring wrap

Handled by the two-segment rule. A strict ring proves why raw restore cannot be
omitted: Pass A future row 1 can overwrite a still-visible S0 historical row.
**PROVEN BY SOURCE**.

### Multiple compressor emits

After counter/frontier restore, each sequential emit overwrites its append row
before incrementing visibility. The induction applies to every emit in the
block. **PROVEN BY SOURCE**.

### Ratio-4 state and mid-cycle starts

The full frontier tensor preserves partial-cycle contents. The restored count
hides appended rows. Sequential per-token updates resume at the exact logical
position and overwrite each emitted index row before scoring can see it.
**PROVEN BY SOURCE**.

### Sparse/indexer already active

Score/top-k row bounds come from restored `layer_n_index_comp`. Historical rows
below that count were never overwritten by append-only Pass A; speculative
rows at or above it are invisible until canonical overwrite. **PROVEN BY
SOURCE**.

### Partial block / partial accept

S0 restoration occurs before the acceptance outcome is replayed. The raw row
set is derived from the whole Pass-A verifier block, not the accepted prefix,
so rejected future rows cannot remain aliased with history. **PROVEN BY
SOURCE**.

### Verifier failure after partial mutation

Once `verifier_may_have_mutated` is true, a failed verifier still requires the
same frontier and raw restoration before any sequential fallback/reference.
The existing caller already refuses fallback if frontier restore fails. C3
must give raw restore the same fail-closed status. **PROVEN BY SOURCE**.

## M. Verdict

### Challenge results

| Current working assumption | Adjudication | Source result |
|---|---|---|
| Attention/index frontier tensors and counters are handled by `spec_frontier_restore()` | CONFIRMED | complete frontier tensors and both counter arrays are copied/restored. **PROVEN BY SOURCE** |
| Target raw-KV physical rows are not fully handled and require explicit pre-Pass-A snapshot/restore | CONFIRMED | frontier functions never copy `layer_raw_cache`; supported strict-ring layouts permit future/history aliasing. **PROVEN BY SOURCE** |
| Compressed-cache append rows can remain stale after counter restore | CONFIRMED | restored counts hide them, and sequential emit overwrites before increment/read. **PROVEN BY SOURCE** |
| Batch scratch / HC contents generally need no restoration | WEAKENED | tensor contents and batch pointer pairs need none for target Pass B, but shared `active_tier` must be restored in a placement/multi-tier extension. **PROVEN BY SOURCE** |
| Support-model state is not mutated by target verification | CONFIRMED | no DSpark/MTP support-cache writer occurs in the generic target verifier call graph; target-hidden capture is proposer input, not support cache. **PROVEN BY SOURCE** |

`C3 SOURCE AUDIT GO — restoration set is source-provable`

### Required before Pass A

- record `start` and the exact forced suffix length;
- snapshot every target layer's raw destination rows
  `(start+t) % raw_cap`, split into two segments on wrap;
- call `spec_frontier_snapshot()`;
- for a future multi-tier version, save `active_tier` and restore the matching
  backend device later.

### Required between Pass A and Pass B

- reset `s->checkpoint.len = start`;
- invalidate verifier-produced DSpark capture metadata;
- require successful `spec_frontier_restore()`;
- restore the saved raw-KV physical rows and require success;
- in a multi-tier version, restore tier/device metadata;
- begin ordinary sequential forced-token decode only after all restoration
  commands complete.

### Proven unnecessary to restore

- stale attention/indexer compressed-cache append rows;
- batch HC/Q/KV/FFN contents and batch HC pointer orientation;
- decode HC contents or pointer identity, because Pass A does not change the
  identity and Pass B overwrites contents;
- `spec_logits`, row-logit/top buffers, prefix-capture scratch, and selection
  scratch;
- DSpark/MTP support-model cache contents, because target Pass A does not write
  them;
- stale checkpoint-vector contents beyond restored logical `len`;
- canonical session logits, because Pass A does not modify them;
- pure statistics/profile fields.

### Remaining unknowns

None that block the single-device Metal C3 restoration set.

The following are outside this gate rather than silently treated as benign:

- runtime validation that the implemented C3 copy schedule is non-perturbing;
- multi-tier/CUDA duplicated-cache implementation details beyond the
  source-proven need to restore tier/device metadata and audit duplicated raw
  caches;
- scheduler behavior if a diagnostic returns to normal proposal scheduling
  without completing ordinary capture/checkpoint-note flow.

Those items belong to later implementation/runtime validation and do not
change the source-provable S0 restoration set for the frozen M4 Metal
first-divergence experiment.
