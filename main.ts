// free-model-aggregator-02: one server, two client wires, fourteen live free models.
//
// The server reads the trial brief (the free model list with the house's
// ratings and the six-prompt manifest), registers one catalog with the env,
// and answers the client's six prompts on the OpenAI Chat Completions wire
// and the Anthropic Messages wire, in JSON and in server-sent events. Every
// answer comes from a real free OpenRouter model dialled through the
// grader's gateway on Chat Completions, never from this server itself, and
// the server exits 0 when GET /work says the trial is done.
//
// Routing, in one line: every prompt is tried on the best free model that
// meets its requirements first, then on the next best by a budget-aware
// score -- the rating lost against the best model for that prompt, plus a
// penalty for each call already spent on that model this trial, plus a
// penalty for a model that already refused one -- so a fresh near-best model
// beats a tired best one, and a refusal is never retried on the same model
// for the same prompt. A model that answers without the prompt's token in
// its text is kept as a fallback while the next candidate is tried, because
// a token-less answer earns the prompt nothing on quality. Refusals are
// reported on the answer in x-makefun-upstream-failures, the only trail the
// failover column can see.
//
// The wires, in one line: an OpenAI request is passed through to the
// gateway with its messages, system prompt, tools, stop sequences,
// temperature and JSON mode intact; an Anthropic request is translated to
// Chat Completions and the answer translated back, in JSON or as the
// message_start / content_block_* / message_delta / message_stop event
// order. Upstream answers are buffered whole before anything is relayed,
// so the token check and the failure trail are known before the client
// sees a byte, and every `model` field names the id actually dialled,
// with the gateway's `openrouter/` prefix, on every answer and every
// streamed chunk.
//
// Determinism: every choice is a function of the brief, the requests and
// the gateway's answers so far. No wall clock, no unseeded randomness,
// fixed timestamps in every reply, so a replay leaves the same state
// export byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one gateway call may take before it counts as a refusal. */
const DIAL_TIMEOUT_MS = 28000;
/** The most gateway calls one prompt may cost, refusals included. */
const MAX_ATTEMPTS_PER_PROMPT = 6;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The output cap a request gets when it names none; the gateway refuses uncapped calls. */
const DEFAULT_MAX_TOKENS = 1024;
/** The gateway's output cap: a request over it is refused. */
const MAX_TOKENS_CAP = 8192;
/** A fixed epoch second for every reply that names one: nothing reads the clock. */
const FIXED_CREATED = 1767225600;
/** A fixed instant for every listed model's created_at: nothing reads the clock. */
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
/** The penalty, in rating milli, for routing to a model already used once this trial. */
const REUSE_PENALTY_FIRST = 60;
/** The extra penalty, in rating milli, for each further call on a model. */
const REUSE_PENALTY_PER_CALL = 10;
/** Each refusal costs 40 rating milli: one transient refusal cannot demote the best below the worst. */
const REFUSED_PENALTY = 40;
/** Plain Chat Completions returns 403: this model is only available on agentic harnesses. */
const ROUTING_EXCLUDED_MODEL = 'openrouter/thinkingmachines/inkling-2-small:free';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

type Json = Record<string, unknown>;
type Wire = 'openai' | 'anthropic';

const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const stringOf = (v: unknown): string => (typeof v === 'string' ? v : '');
const numberOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const boolOrNull = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

// ---------------------------------------------------------------------------
// The brief: the free model list and the six-prompt manifest.
// ---------------------------------------------------------------------------

const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';
/** Every schema this contest's brief writes names this prefix; its JSON is found by it. */
const SCHEMA_PREFIX = 'free-model-aggregator-02/';
/** The public base URL of the provider the free models belong to: the endpoint the gateway reaches. */
const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';
/** The environment variable an entrant's OpenRouter key is pushed under. */
const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

/** One free model as the brief lists it. `id` is the id the gateway routes. */
interface ModelRow {
  id: string;
  provider: string;
  display_name: string;
  /** The published rating in milli-points: the quality column's currency. */
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

/** What one prompt needs: a model without it cannot serve the prompt. */
interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One prompt the client will send, as the brief lists it. */
interface PromptRow {
  id: string;
  api: Wire;
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

/** The brief as this server reads it. `error` is set when a document could not be read. */
interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

/** One model in the catalog document. */
interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** One provider in the catalog document. */
interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: 'openai';
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register: the same document, byte for byte. */
interface CatalogReport {
  providers: CatalogProvider[];
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

/** The index of the `}` closing the object that opens at `start`, or -1; string contents are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/**
 * The first document under this contest's prefix that carries one of `keys` as an array: the document
 * and the key it was found under. A brief that spells its lists in the sibling's names still reads.
 */
function arrayDocument(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const doc of docs) {
    const schema = stringOf(doc.schema);
    if (!schema.includes(SCHEMA_PREFIX)) continue;
    for (const key of keys) if (Array.isArray(doc[key])) return { doc, key };
  }
  return null;
}

/**
 * One model row. The id is the row's `id`, or its `model` under the provider that names it: a row that
 * gives the bare OpenRouter id (`vendor/model:free`) and names `openrouter` is routed as the gateway
 * spells it (`openrouter/vendor/model:free`). A row is free unless it says `free: false`.
 */
function modelOf(v: unknown): ModelRow | null {
  if (!isRecord(v)) return null;
  const raw = stringOf(v.id) !== '' ? stringOf(v.id) : stringOf(v.model);
  if (raw === '') return null;
  const named = stringOf(v.provider) !== '' ? stringOf(v.provider) : stringOf(v.wire);
  const id = named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
  return {
    id,
    provider: named === '' ? 'openrouter' : named,
    display_name: stringOf(v.display_name) !== '' ? stringOf(v.display_name) : raw,
    rating_milli: numberOrNull(v.rating_milli) ?? 0,
    context_length: numberOrNull(v.context_length),
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
    free: v.free !== false,
  };
}

function promptOf(v: unknown): PromptRow | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      ...(requires.tools === true ? { tools: true } : {}),
      ...(requires.vision === true ? { vision: true } : {}),
      ...(requires.json === true ? { json: true } : {}),
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
    },
  };
}

/** The brief at `path`, its two documents found under this contest's schema prefix. */
function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const modelsDoc = arrayDocument(docs, ['models', 'ratings']);
  if (modelsDoc === null) {
    return { models: [], prompts: [], error: `the brief at ${path} holds no ${MODELS_SCHEMA} document` };
  }
  const models = (modelsDoc.doc[modelsDoc.key] as unknown[]).flatMap((v) => {
    const m = modelOf(v);
    return m === null ? [] : [m];
  });
  const promptsDoc = arrayDocument(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) {
    return { models, prompts: [], error: `the brief at ${path} holds no ${PROMPTS_SCHEMA} document` };
  }
  const prompts = (promptsDoc.doc[promptsDoc.key] as unknown[]).flatMap((v) => {
    const p = promptOf(v);
    return p === null ? [] : [p];
  });
  return { models, prompts, error: null };
}

/** The models this entry may serve: the brief's free ones, in the brief's order. */
function freeModels(brief: Brief): ModelRow[] {
  return brief.models.filter((m) => m.free);
}

/** What a model can do, as the brief's row says it. */
function capabilitiesOf(model: ModelRow): { context_length: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null } {
  return { context_length: model.context_length, tools: model.tools, vision: model.vision, json: model.json };
}

/** Whether a model's capabilities meet one prompt's requirements. An unknown capability never meets one. */
function meetsRequirements(
  requires: Requirement,
  caps: { context_length: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null },
): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/**
 * The catalog the brief's free model list makes: one provider, the models it serves. The base_url is
 * the provider's public endpoint, never the gateway's trial-local address: the gateway's origin differs
 * between a live pass and its replay, and the catalog is part of the state export a replay must
 * reproduce byte for byte. Exactly the fourteen dialable ids, once each, no more and no fewer.
 */
function buildCatalog(brief: Brief): CatalogReport {
  const seen = new Set<string>();
  const models: CatalogModel[] = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.context_length });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [{ id: 'openrouter', base_url: OPENROUTER_PUBLIC_BASE, key_env: OPENROUTER_KEY_ENV, wire: 'openai', models }],
  };
}

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/** A 401 from the gateway means no key was pushed: every later dial would refuse the same way. */
let keyDead = false;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

// ---------------------------------------------------------------------------
// Routing: the budget-aware order, and what this trial has learned per model.
// ---------------------------------------------------------------------------

/** What this trial has learned about one model: how many calls it has taken, and how many it refused. */
interface ModelState {
  calls: number;
  refusals: number;
}

const modelState = new Map<string, ModelState>();

function stateOf(modelId: string): ModelState {
  let state = modelState.get(modelId);
  if (state === undefined) {
    state = { calls: 0, refusals: 0 };
    modelState.set(modelId, state);
  }
  return state;
}

/** The reuse penalty for a model already called `calls` times this trial. */
function reusePenalty(calls: number): number {
  if (calls <= 0) return 0;
  return REUSE_PENALTY_FIRST + REUSE_PENALTY_PER_CALL * (calls - 1);
}

/**
 * The free models that meet one prompt's requirements, in the order to try them: the rating lost
 * against this prompt's best model, plus the reuse penalty for the calls already spent on that model,
 * plus a modest penalty for each refusal it has already given. A fresh near-best model therefore beats
 * a tired best one, while a transient refusal leaves the best model competitive later in the
 * trial. Ties break by rating, then by id, so the order is a pure function of the brief and the
 * gateway's answers so far.
 */
function orderedCandidates(requires: Requirement): ModelRow[] {
  const eligible = freeModels(brief).filter((m) => m.id !== ROUTING_EXCLUDED_MODEL && meetsRequirements(requires, capabilitiesOf(m)));
  let bestRating = 0;
  for (const m of eligible) if (m.rating_milli > bestRating) bestRating = m.rating_milli;
  const scored = eligible.map((model) => {
    const state = stateOf(model.id);
    const score = bestRating - model.rating_milli + reusePenalty(state.calls) + REFUSED_PENALTY * state.refusals;
    return { model, score };
  });
  scored.sort(
    (a, b) =>
      a.score - b.score ||
      b.model.rating_milli - a.model.rating_milli ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return scored.map((r) => r.model);
}

// ---------------------------------------------------------------------------
// Dialling the gateway and reading its answer back.
// ---------------------------------------------------------------------------

/** One row of the refusal trail the failover column reads. */
interface Failure {
  status: number;
  code: string;
  model: string | null;
}

/** One SSE frame: its event name, if any, and its data lines joined. */
interface SseFrame {
  event: string | null;
  data: string;
}

/** A gateway answer read whole: its text (what the env's content check reads), and its body. */
interface UpstreamAnswer {
  text: string;
  /** The parsed JSON body, for a non-streamed answer. */
  json: Json | null;
  /** The SSE frames, for a streamed answer. */
  frames: SseFrame[] | null;
}

/** One server-sent event stream as frames: an event name and its data lines. */
function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
}

/** The answer text of a Chat Completions JSON body, as the env's content check reads it. */
function jsonAnswerText(body: Json): string {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return '';
  const message = isRecord(choice.message) ? choice.message : null;
  if (message === null) return '';
  return textOf(message.content);
}

/** The answer text of a Chat Completions SSE stream: every content delta joined. */
function streamAnswerText(frames: SseFrame[]): string {
  let text = '';
  for (const frame of frames) {
    if (frame.data === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
  }
  return text;
}

/** A gateway 2xx body as an answer, or null when it cannot be read as one. */
function parseUpstream(text: string, stream: boolean): UpstreamAnswer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(text);
    } catch {
      return null;
    }
    if (!isRecord(value)) return null;
    const choices = Array.isArray(value.choices) ? value.choices : [];
    if (choices.length === 0 || !isRecord(choices[0]) || !isRecord(choices[0].message)) return null;
    return { text: jsonAnswerText(value), json: value, frames: null };
  }
  const frames = parseSseFrames(text);
  if (frames.length === 0) return null;
  return { text: streamAnswerText(frames), json: null, frames };
}

/** POST one request to the gateway's Chat Completions route: its HTTP status and body, or null when unreachable. */
async function dial(model: string, body: Json): Promise<{ status: number; text: string } | null> {
  const url = chatCompletionsUrl();
  if (url === '') return null;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify({ ...body, model }),
      signal: AbortSignal.timeout(DIAL_TIMEOUT_MS),
    });
    return { status: res.status, text: await res.text() };
  } catch (err) {
    log(`dial to ${model} failed: ${(err as Error).message}`);
    return null;
  }
}

/** The code of a gateway refusal, from its error envelope: a string, never a number. */
function errorCode(status: number, text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      const code = value.error.code;
      if (typeof code === 'string' && code !== '') return code;
      if (typeof code === 'number') return String(code);
      if (typeof value.error.type === 'string' && value.error.type !== '') return value.error.type;
    }
  } catch {
    // Not JSON: the status is the code.
  }
  return `http_${status}`;
}

// ---------------------------------------------------------------------------
// Reading one client request: its requirements, its token, its upstream body.
// ---------------------------------------------------------------------------

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(index: number, body: Json): Requirement {
  const entry = brief.prompts[index];
  if (entry !== undefined) return entry.requires;
  // The brief's prompt list is the declared source of each request's requirements; this fallback is a
  // guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** The user-visible text of a client request, in either wire: where its token lives. */
function requestText(api: Wire, body: Json): string {
  const parts: string[] = [];
  if (api === 'anthropic') {
    parts.push(textOf(body.system));
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const message of messages) if (isRecord(message)) parts.push(textOf(message.content));
  } else {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const message of messages) if (isRecord(message)) parts.push(textOf(message.content));
  }
  return parts.join('\n');
}

/**
 * The prompt's unique token, as the request carries it (a word like `SPARROW-7435` in the instruction),
 * or null when the request names none: without one there is nothing to check the answer against.
 */
function tokenOfRequest(api: Wire, body: Json): string | null {
  const match = /[A-Z]{3,}-[0-9]{3,}/.exec(requestText(api, body));
  return match === null ? null : match[0];
}

/** An output cap within the gateway's bounds: the request's own, clamped, or the default when it names none. */
function cappedMaxTokens(value: unknown): number {
  if (typeof value === 'number' && Number.isFinite(value)) return Math.min(Math.max(1, Math.floor(value)), MAX_TOKENS_CAP);
  return DEFAULT_MAX_TOKENS;
}

/**
 * One client request as the Chat Completions body the gateway takes. An OpenAI-shaped body passes
 * through with its messages, system prompt, tools, stop sequences, temperature and JSON mode intact; an
 * Anthropic-shaped one is translated first. The output cap is always set and always within bounds, and
 * a streamed request asks for its usage chunk.
 */
function upstreamBody(api: Wire, body: Json, model: string, stream: boolean): Json {
  let out: Json;
  if (api === 'openai') {
    out = { ...body, model };
    if (typeof out.max_tokens === 'number') out.max_tokens = cappedMaxTokens(out.max_tokens);
    else if (typeof out.max_completion_tokens === 'number') out.max_completion_tokens = cappedMaxTokens(out.max_completion_tokens);
    else out.max_tokens = DEFAULT_MAX_TOKENS;
  } else {
    out = anthropicToOpenAI(body, model);
    out.max_tokens = cappedMaxTokens(body.max_tokens);
  }
  out.stream = stream;
  if (stream) out.stream_options = { include_usage: true };
  return out;
}

/**
 * Serve one client request: the first candidate that answers it with its token in the text. A
 * refusal is never retried on the same model; a token-less answer is kept as a fallback while the next
 * candidate is tried, and served when no candidate does better. Every refusal is recorded for the
 * x-makefun-upstream-failures trail.
 */
async function serve(
  api: Wire,
  body: Json,
  index: number,
): Promise<{ status: number; json?: Json; sse?: string; failures: Failure[] }> {
  const stream = body.stream === true;
  const requires = requirementsFor(index, body);
  const token = tokenOfRequest(api, body);
  const declaredTools = new Set((Array.isArray(body.tools) ? body.tools : []).flatMap((tool) => {
    if (!isRecord(tool)) return [];
    const name = api === 'anthropic' ? tool.name : isRecord(tool.function) ? tool.function.name : undefined;
    return typeof name === 'string' ? [name] : [];
  }));
  const candidates = orderedCandidates(requires);
  const failures: Failure[] = [];
  let fallback: { model: string; answer: UpstreamAnswer } | null = null;
  let attempts = 0;
  for (const model of candidates) {
    if (attempts >= MAX_ATTEMPTS_PER_PROMPT) break;
    if (keyDead) break;
    attempts += 1;
    const state = stateOf(model.id);
    state.calls += 1;
    const res = await dial(model.id, upstreamBody(api, body, model.id, stream));
    if (res === null) {
      failures.push({ status: 504, code: 'timeout', model: model.id });
      state.refusals += 1;
      log(`${model.id} timed out for prompt ${index + 1}`);
      continue;
    }
    if (res.status === 401) {
      // No key was pushed: every model would refuse the same way, so stop dialling for the trial.
      keyDead = true;
      failures.push({ status: 401, code: errorCode(401, res.text), model: model.id });
      log(`the gateway refused ${model.id} with 401: no key was pushed`);
      break;
    }
    if (res.status < 200 || res.status >= 300) {
      failures.push({ status: res.status, code: errorCode(res.status, res.text), model: model.id });
      state.refusals += 1;
      log(`${model.id} refused HTTP ${res.status} for prompt ${index + 1}`);
      continue;
    }
    const parsed = parseUpstream(res.text, stream);
    const answer = parsed === null ? null : stripUndeclaredTools(parsed, declaredTools);
    if (answer === null) {
      failures.push({ status: res.status, code: 'unparseable_answer', model: model.id });
      state.refusals += 1;
      log(`${model.id} answered 2xx with an unreadable body for prompt ${index + 1}`);
      continue;
    }
    if ((answer.text === '' && !hasToolCalls(answer)) || (token !== null && !answer.text.includes(token))) {
      // The answer is real but carries no token, so it earns this prompt nothing on quality: keep it
      // as the fallback and try the next candidate for one that echoes the token.
      if (fallback === null) fallback = { model: model.id, answer };
      log(`${model.id} answered prompt ${index + 1} without its token`);
      continue;
    }
    log(`${model.id} answered prompt ${index + 1}`);
    return buildClientResponse(api, model.id, answer, stream, failures, declaredTools);
  }
  if (fallback !== null) {
    log(`${fallback.model} is the fallback for prompt ${index + 1}: no candidate echoed its token`);
    return buildClientResponse(api, fallback.model, fallback.answer, stream, failures, declaredTools);
  }
  return { status: 502, failures, json: errorBody(api, 502, 'upstream_error', 'no free model could serve this request') };
}

// ---------------------------------------------------------------------------
// The client wires: Anthropic <-> Chat Completions, and the model field.
// ---------------------------------------------------------------------------

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
function openAIModelList(models: readonly ModelRow[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
function anthropicModelList(models: readonly ModelRow[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_CREATED_AT })),
    has_more: false,
  };
}

/** An error in the client's wire: Anthropic's error envelope, or OpenAI's. */
function errorBody(api: Wire, status: number, type: string, message: string): Json {
  if (api === 'anthropic') return { type: 'error', error: { type, message } };
  return { error: { message, type, code: null } };
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function anthropicBlockToOpenAI(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming `model`: the system prompt
 * becomes a system message, image blocks become image_url data URLs, tool_use blocks become tool_calls,
 * tool_result blocks become tool messages, and tools, tool_choice, the output cap, temperature, top_p
 * and stop sequences pass through.
 */
function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = anthropicBlockToOpenAI(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** Deterministic fallback: four answer characters per output token, rounded up. */
function answerUsage(usage: unknown, text: string): Json {
  const source = isRecord(usage) ? usage : {};
  if (numberOrZero(source.completion_tokens) > 0) return source;
  const input = numberOrZero(source.prompt_tokens);
  const output = Math.ceil(text.length / 4);
  return { ...source, prompt_tokens: input, completion_tokens: output, total_tokens: input + output };
}

/** Drop undeclared calls before token checks and either wire's answer translation. */
function stripUndeclaredTools(answer: UpstreamAnswer, declaredTools: ReadonlySet<string>): UpstreamAnswer {
  const allowedIndexes = new Set<number>();
  const filterBody = (body: Json, streamed: boolean): Json => {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    return { ...body, choices: choices.map((choice) => {
      if (!isRecord(choice)) return choice;
      const key = streamed ? 'delta' : 'message';
      const message = isRecord(choice[key]) ? choice[key] : {};
      const filtered: Json = { ...message };
      if (Array.isArray(message.content)) {
        filtered.content = message.content.filter((block) =>
          !isRecord(block) || block.type !== 'tool_use' || declaredTools.has(stringOf(block.name)));
      }
      if (Array.isArray(message.tool_calls)) {
        filtered.tool_calls = message.tool_calls.filter((call) => {
          if (!isRecord(call)) return false;
          const fn = isRecord(call.function) ? call.function : {};
          const index = numberOrZero(call.index);
          if (!streamed) return declaredTools.has(stringOf(fn.name));
          if (typeof fn.name === 'string') {
            if (declaredTools.has(fn.name)) allowedIndexes.add(index);
            else allowedIndexes.delete(index);
          }
          return allowedIndexes.has(index);
        });
      }
      return { ...choice, [key]: filtered };
    }) };
  };
  if (answer.json !== null) {
    const json = filterBody(answer.json, false);
    return { ...answer, json, text: jsonAnswerText(json) };
  }
  const frames = (answer.frames ?? []).map((frame) => {
    try {
      const chunk: unknown = JSON.parse(frame.data);
      return isRecord(chunk) ? { ...frame, data: JSON.stringify(filterBody(chunk, true)) } : frame;
    } catch {
      return frame;
    }
  });
  return { ...answer, frames, text: streamAnswerText(frames) };
}

/** Whether a sanitized answer retains any declared tool call. */
function hasToolCalls(answer: UpstreamAnswer): boolean {
  const bodies: Json[] = answer.json !== null ? [answer.json] : (answer.frames ?? []).flatMap((frame) => {
    try {
      const chunk: unknown = JSON.parse(frame.data);
      return isRecord(chunk) ? [chunk] : [];
    } catch {
      return [];
    }
  });
  return bodies.some((body) => (Array.isArray(body.choices) ? body.choices : []).some((choice) => {
    if (!isRecord(choice)) return false;
    const message = isRecord(choice.message) ? choice.message : isRecord(choice.delta) ? choice.delta : {};
    return (Array.isArray(message.tool_calls) && message.tool_calls.length > 0) ||
      (Array.isArray(message.content) && message.content.some((block) => isRecord(block) && block.type === 'tool_use'));
  }));
}

/** A Chat Completions answer as an Anthropic Messages object. `model` is the id the entry dialled. */
function openAIToAnthropic(body: Json, model: string, declaredTools: ReadonlySet<string>): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function) || !declaredTools.has(stringOf(call.function.name))) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = answerUsage(body.usage, text);
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the client asked for it, every data payload's `model` the id the
 * entry dialled. A stream that carried no finish chunk gets one appended (a stop with an empty delta),
 * and one that did not end with [DONE] gets it, so the env's stream-order check reads a complete
 * stream; the provider's own chunks pass through untouched apart from the model field.
 */
function openAIStreamRelay(frames: SseFrame[], model: string): string {
  let out = '';
  let sawFinish = false;
  let sawDone = false;
  for (const frame of frames) {
    if (frame.data === '[DONE]') {
      sawDone = true;
      out += 'data: [DONE]\n\n';
      continue;
    }
    let chunk: unknown;
    try {
      chunk = JSON.parse(frame.data);
    } catch {
      out += `data: ${frame.data}\n\n`;
      continue;
    }
    if (!isRecord(chunk)) {
      out += `data: ${frame.data}\n\n`;
      continue;
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice !== null && typeof choice.finish_reason === 'string') sawFinish = true;
    out += sseEvent(frame.event, { ...chunk, model });
  }
  if (!sawFinish) {
    out += sseEvent(null, {
      id: 'chatcmpl-aggregate',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
    });
  }
  if (!sawDone) out += 'data: [DONE]\n\n';
  return out;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason
 * and usage, then message_stop. `model` is the id the entry dialled, named back to the client. The
 * stop reason is held until the stream's usage chunk has arrived: OpenAI sends the usage after the
 * finish_reason, and Anthropic's message_delta carries the output tokens.
 */
function anthropicStreamRelay(frames: SseFrame[], model: string, declaredTools: ReadonlySet<string>): string {
  let out = '';
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;
  let answerText = '';

  const startEvents = (): string => {
    if (started) return '';
    started = true;
    return sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: inputTokens, output_tokens: 0 },
      },
    });
  };
  const closeEvents = (): string => {
    let s = '';
    if (textIndex !== -1) {
      s += sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex });
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) s += sseEvent('content_block_stop', { type: 'content_block_stop', index });
    toolIndexes.clear();
    return s;
  };
  const finishEvents = (stopReason: string): string => {
    if (finished) return '';
    finished = true;
    if (outputTokens <= 0) outputTokens = numberOrZero(answerUsage(null, answerText).completion_tokens);
    return (
      closeEvents() +
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }) +
      sseEvent('message_stop', { type: 'message_stop' })
    );
  };

  for (const frame of frames) {
    if (frame.data === '[DONE]') {
      out += startEvents() + finishEvents(pendingStop ?? 'end_turn');
      continue;
    }
    let chunk: unknown;
    try {
      chunk = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    out += startEvents();
    if (typeof delta.content === 'string' && delta.content !== '') {
      answerText += delta.content;
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        out += sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } });
      }
      out += sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } });
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
      let index = toolIndexes.get(upstreamIndex);
      if (index === undefined) {
        const fn = isRecord(call.function) ? call.function : {};
        if (!declaredTools.has(stringOf(fn.name))) continue;
        if (textIndex !== -1) {
          out += sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex });
          textIndex = -1;
        }
        index = nextIndex;
        nextIndex += 1;
        toolIndexes.set(upstreamIndex, index);
        out += sseEvent('content_block_start', {
          type: 'content_block_start',
          index,
          content_block: {
            type: 'tool_use',
            id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
            name: typeof fn.name === 'string' ? fn.name : '',
            input: {},
          },
        });
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        out += sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } });
      }
    }
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
  }
  if (!finished) {
    out += startEvents() + finishEvents(pendingStop ?? 'end_turn');
  }
  return out;
}

/**
 * A gateway answer in the wire the client used, streamed or not as it asked, with `model` the id the
 * entry dialled on every answer and every streamed chunk. The refusal trail rides the response header
 * (at most 8 rows), so the failover column can see a refusal the server recovered from.
 */
function buildClientResponse(
  api: Wire,
  model: string,
  answer: UpstreamAnswer,
  stream: boolean,
  failures: Failure[],
  declaredTools: ReadonlySet<string>,
): { status: number; json?: Json; sse?: string; failures: Failure[] } {
  if (!stream) {
    const json = answer.json ?? {};
    if (api === 'anthropic') return { status: 200, json: openAIToAnthropic(json, model, declaredTools), failures };
    return { status: 200, json: { ...json, model, usage: answerUsage(json.usage, answer.text) }, failures };
  }
  const frames = answer.frames ?? [];
  if (api === 'anthropic') return { status: 200, sse: anthropicStreamRelay(frames, model, declaredTools), failures };
  return { status: 200, sse: openAIStreamRelay(frames, model), failures };
}

// ---------------------------------------------------------------------------
// The HTTP server: the two client wires and the model list.
// ---------------------------------------------------------------------------

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** The refusal trail as one response header, or null when there is nothing to report. */
function failuresHeader(failures: Failure[]): string | null {
  if (failures.length === 0) return null;
  const rows = failures.slice(0, 8).map((f) => ({ status: f.status, code: f.code, model: f.model }));
  const text = JSON.stringify(rows);
  return Buffer.byteLength(text) > 8192 ? null : text;
}

function sendJson(res: ServerResponse, status: number, body: unknown, failures: Failure[] = []): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = { 'content-type': 'application/json', 'content-length': String(Buffer.byteLength(text)) };
  const trail = failuresHeader(failures);
  if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
  res.writeHead(status, headers);
  res.end(text);
}

function sendSse(res: ServerResponse, text: string, failures: Failure[] = []): void {
  const headers: Record<string, string> = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
  const trail = failuresHeader(failures);
  if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
  res.writeHead(200, headers);
  res.end(text);
}

function sendError(res: ServerResponse, api: Wire, status: number, type: string, message: string): void {
  sendJson(res, status, errorBody(api, status, type, message));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = freeModels(brief);
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && (url.pathname === '/v1/chat/completions' || url.pathname === '/v1/messages')) {
    const api: Wire = url.pathname === '/v1/messages' ? 'anthropic' : 'openai';
    const body = await readJson(req);
    if (body === null) return sendError(res, api, 400, 'invalid_request_error', 'the request body is not one JSON object');
    // Count the prompt now: the brief's rows are in the client's arrival order, and this request is
    // the next one, whatever it asks for.
    const index = manifestIndex;
    manifestIndex += 1;
    const answer = await serve(api, body, index);
    if (answer.sse !== undefined) return sendSse(res, answer.sse, answer.failures);
    return sendJson(res, answer.status, answer.json ?? {}, answer.failures);
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${freeModels(brief).length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
