// ╔══════════════════════════════════════════════════════════════════╗
// ║       IndexGen Pro — server.js v5.1 (3-PASS EXTRACTION)        ║
// ╠══════════════════════════════════════════════════════════════════╣
// ║  FIX: 3-pass system — no page is ever silently skipped          ║
// ║  Pass 1 : Parallel (fast, catches ~90% of pages)               ║
// ║  Pass 2 : Sequential with 8s gaps (rate-limit safe)            ║
// ║  Pass 3 : Page-by-page nuclear option (last resort)            ║
// ║  Only reports skip if ALL 3 passes fail for a page             ║
// ╚══════════════════════════════════════════════════════════════════╝

import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import { exec } from 'child_process';
import { promisify } from 'util';
import fs from 'fs';
import path from 'path';
import os from 'os';
import { GoogleGenAI } from '@google/genai';
import cors from 'cors';

const execAsync = promisify(exec);
const app = express();
app.use(cors());
app.use(express.json());

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// ─── CONFIG ───────────────────────────────────────────────────────────────────
const MODEL_NAME   = 'gemini-2.5-flash';
const ONE_PAGE_MAX = 25;
const CHUNK_SIZE   = 5;   // ↓ from 10 → smaller = fewer rate limit hits
const CONCURRENT   = 2;   // ↓ from 3 → safer for free tier (10 RPM)
const MAX_RETRIES  = 5;   // ↑ more retries per chunk
const MAX_FILE_MB  = 50;
const JOB_TTL_MS   = 20 * 60 * 1000;

if (!process.env.GEMINI_API_KEY) {
  console.error('FATAL: GEMINI_API_KEY not set.');
  process.exit(1);
}

// ─── IN-MEMORY JOB STORE ─────────────────────────────────────────────────────
const jobs = new Map();

// ─── MULTER ───────────────────────────────────────────────────────────────────
const upload = multer({
  dest: os.tmpdir(),
  limits: { fileSize: MAX_FILE_MB * 1024 * 1024 },
  fileFilter: (_, file, cb) =>
    file.mimetype === 'application/pdf'
      ? cb(null, true)
      : cb(new Error(`Only PDF files allowed. Received: ${file.mimetype}`))
});

// ─── SCHEMAS ──────────────────────────────────────────────────────────────────
const INDEX_SCHEMA = {
  type: 'array',
  items: {
    type: 'object',
    properties: {
      page:  { type: 'integer' },
      topic: { type: 'string'  },
      level: { type: 'integer' }
    },
    required: ['page', 'topic', 'level']
  }
};

const CONTEXT_SCHEMA = {
  type: 'object',
  properties: {
    subject:        { type: 'string'  },
    documentType:   { type: 'string'  },
    theme:          { type: 'string'  },
    headingPattern: { type: 'string'  },
    entryFormat:    { type: 'string'  },
    isHandwritten:  { type: 'boolean' },
    topicStructure: { type: 'string'  },
    language:       { type: 'string'  }
  },
  required: [
    'subject','documentType','theme',
    'headingPattern','entryFormat',
    'isHandwritten','topicStructure','language'
  ]
};

// ─── ROUTES ───────────────────────────────────────────────────────────────────
app.get('/', (_, res) => res.json({ status: 'IndexGen Pro v5.1', model: MODEL_NAME }));

app.get('/health', (_, res) => res.json({
  ok: true, model: MODEL_NAME, activeJobs: jobs.size,
  config: { chunkSize: CHUNK_SIZE, concurrent: CONCURRENT, onPageMax: ONE_PAGE_MAX }
}));

// ─── UPLOAD ENDPOINT (returns jobId in <1s) ───────────────────────────────────
app.post('/api/generate-index', upload.single('pdf'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No PDF uploaded. Field name must be "pdf".' });
  }

  const jobId = `job_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

  jobs.set(jobId, {
    status: 'queued', progress: '📋 Job queued...', step: 0,
    pagesTotal: 0, pagesDone: 0,
    result: null, error: null, startedAt: Date.now()
  });

  res.json({ jobId, status: 'queued' });

  processJob(jobId, req.file).catch(err => {
    const job = jobs.get(jobId);
    if (job) { job.status = 'error'; job.error = err.message; }
    console.error(`[${jobId}] Unhandled:`, err.message);
  });
});

// ─── POLLING ENDPOINT ─────────────────────────────────────────────────────────
app.get('/api/status/:jobId', (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job) {
    return res.status(404).json({
      error: 'Job not found. Server may have restarted. Please re-upload your PDF.'
    });
  }
  res.json({
    status: job.status, progress: job.progress, step: job.step,
    pagesTotal: job.pagesTotal, pagesDone: job.pagesDone,
    result: job.result, error: job.error,
    elapsedMs: Date.now() - job.startedAt
  });
});

// ─── MAIN BACKGROUND PROCESSOR ───────────────────────────────────────────────
async function processJob(jobId, file) {
  const job     = jobs.get(jobId);
  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), 'idx-'));
  const pdfPath = file.path;
  const t0      = Date.now();

  try {
    // ── STEP 1: PDF → JPEG ────────────────────────────────────────────────────
    job.status = 'processing'; job.step = 1;
    job.progress = '⚙️  Step 1/4 — Converting PDF to images...';

    await execAsync(`pdftoppm -jpeg -r 150 "${pdfPath}" "${workDir}/page"`);

    const pageFiles = fs.readdirSync(workDir)
      .filter(f => f.endsWith('.jpg'))
      .sort((a, b) => {
        const na = parseInt(a.match(/\d+/)?.[0] || 0);
        const nb = parseInt(b.match(/\d+/)?.[0] || 0);
        return na - nb;
      });

    if (!pageFiles.length) {
      throw new Error('PDF conversion failed — file may be corrupt or password-protected.');
    }

    job.pagesTotal = pageFiles.length;
    job.step = 2;
    job.progress = `📄 Step 2/4 — Analyzing document (${pageFiles.length} pages found)...`;
    console.log(`\n[${jobId}] ${pageFiles.length} pages (${elapsed(t0)})`);

    // ── STEP 2: Context analysis ───────────────────────────────────────────────
    const sampleFiles = pageFiles.slice(0, Math.min(3, pageFiles.length));
    const docCtx      = await analyzeContext(workDir, sampleFiles, pageFiles.length);
    console.log(`[${jobId}] "${docCtx.theme}" | ${docCtx.topicStructure} (${elapsed(t0)})`);

    // ── STEP 3: 3-Pass extraction ──────────────────────────────────────────────
    job.step = 3;
    job.progress = `🔍 Step 3/4 — Extracting headings from all ${pageFiles.length} pages...`;

    const { allEntries, failed } = await runAllChunks(jobId, workDir, pageFiles, docCtx, job, t0);

    // Sort + dedup + sanitize
    let entries = [...allEntries].sort((a, b) => a.page - b.page);
    const seen  = new Set();
    entries = entries.filter(({ page, topic }) => {
      const key = `${page}|||${topic.toLowerCase().trim()}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
    entries = entries.map(e => ({
      ...e, topic: e.topic.replace(/<(?!\/?em>)[^>]+>/gi, '').trim()
    }));

    console.log(`[${jobId}] ${entries.length} unique entries (${elapsed(t0)})`);

    // ── STEP 4: Consolidate ───────────────────────────────────────────────────
    job.step = 4;
    let finalEntries = entries;
    let consolidated = false;

    if (entries.length > ONE_PAGE_MAX) {
      job.progress = `✂️  Step 4/4 — Selecting best ${ONE_PAGE_MAX} from ${entries.length}...`;
      finalEntries = await consolidate(entries, docCtx);
      consolidated = true;
    } else {
      job.progress = `✅ Step 4/4 — ${entries.length} entries fit on one page.`;
    }

    const totalMs = Date.now() - t0;
    console.log(`[${jobId}] ✅ Done in ${(totalMs/1000).toFixed(1)}s | ${finalEntries.length} entries\n`);

    job.status   = 'done';
    job.progress = `✅ Complete — ${finalEntries.length} entries in ${(totalMs/1000).toFixed(1)}s`;
    job.result = {
      success: true,
      totalPages: pageFiles.length,
      rawEntries: entries.length,
      finalEntries: finalEntries.length,
      consolidated,
      processingMs: totalMs,
      documentContext: docCtx,
      index: finalEntries,
      // Only warn if pages permanently failed all 3 passes
      ...(failed.length > 0 ? {
        warning: `⚠️ ${failed.length} page(s) could not be read after 3 attempts: ${failed.map(f => f.page).join(', ')}`
      } : {})
    };

    setTimeout(() => { jobs.delete(jobId); }, JOB_TTL_MS);

  } catch (err) {
    console.error(`[${jobId}] Fatal:`, err.message);
    job.status  = 'error';
    job.error   = err.message || 'Internal error. Please try again.';
    setTimeout(() => jobs.delete(jobId), 5 * 60 * 1000);
  } finally {
    fs.rmSync(workDir, { recursive: true, force: true });
    fs.unlink(pdfPath, () => {});
  }
}

// ─── 3-PASS EXTRACTION ENGINE ─────────────────────────────────────────────────
async function runAllChunks(jobId, workDir, pageFiles, docCtx, job, t0) {
  const chunkTasks = [];
  for (let i = 0; i < pageFiles.length; i += CHUNK_SIZE) {
    chunkTasks.push({ files: pageFiles.slice(i, i + CHUNK_SIZE), offset: i });
  }

  let allEntries    = [];
  let pass1Failed   = [];
  let processedPgs  = 0;

  // ════════════════════════════════════════════════════════════════════
  // PASS 1 — Parallel (fast path, catches ~90% of chunks)
  // ════════════════════════════════════════════════════════════════════
  console.log(`[${jobId}] PASS 1: ${chunkTasks.length} chunks (CONCURRENT=${CONCURRENT})`);

  for (let b = 0; b < chunkTasks.length; b += CONCURRENT) {
    const batch   = chunkTasks.slice(b, b + CONCURRENT);
    const settled = await Promise.allSettled(
      batch.map(t => extractWithSmartRetry(workDir, t.files, t.offset, docCtx, MAX_RETRIES))
    );

    settled.forEach((result, idx) => {
      const task = batch[idx];
      if (result.status === 'fulfilled') {
        allEntries.push(...result.value);
        console.log(`[${jobId}] P1 ✓ pages ${task.offset+1}–${task.offset+task.files.length}`);
      } else {
        pass1Failed.push(task);
        console.warn(`[${jobId}] P1 ✗ pages ${task.offset+1}–${task.offset+task.files.length}: ${result.reason?.message?.slice(0,80)}`);
      }
      processedPgs += task.files.length;
    });

    job.pagesDone = processedPgs;
    job.progress  = `🔍 Pass 1/3 — Scanned ${processedPgs}/${pageFiles.length} pages (${allEntries.length} headings found)...`;

    // Rate-limit guard between parallel batches
    if (b + CONCURRENT < chunkTasks.length) await sleep(1200);
  }

  if (pass1Failed.length === 0) {
    console.log(`[${jobId}] ✅ All chunks succeeded on Pass 1!`);
    return { allEntries, failed: [] };
  }

  // ════════════════════════════════════════════════════════════════════
  // PASS 2 — Sequential retry (rate-limit safe, 8s between chunks)
  // ════════════════════════════════════════════════════════════════════
  console.log(`[${jobId}] PASS 2: ${pass1Failed.length} failed chunks → sequential retry (8s gaps)`);
  job.progress = `🔄 Pass 2/3 — Retrying ${pass1Failed.length} failed page ranges (waiting for rate limit to clear)...`;

  await sleep(15000); // Let rate limits fully reset first

  let pass2Failed = [];
  for (const task of pass1Failed) {
    try {
      const entries = await extractWithSmartRetry(workDir, task.files, task.offset, docCtx, 3);
      allEntries.push(...entries);
      console.log(`[${jobId}] P2 ✓ pages ${task.offset+1}–${task.offset+task.files.length}`);
    } catch (err) {
      pass2Failed.push(task);
      console.warn(`[${jobId}] P2 ✗ pages ${task.offset+1}–${task.offset+task.files.length}`);
    }
    // Always wait between sequential requests in Pass 2
    await sleep(8000);
  }

  if (pass2Failed.length === 0) {
    console.log(`[${jobId}] ✅ All remaining chunks succeeded on Pass 2!`);
    return { allEntries, failed: [] };
  }

  // ════════════════════════════════════════════════════════════════════
  // PASS 3 — Page-by-page nuclear option (last resort)
  // ════════════════════════════════════════════════════════════════════
  console.log(`[${jobId}] PASS 3: ${pass2Failed.length} chunks → page-by-page (5s between each)`);
  job.progress = `🔄 Pass 3/3 — Page-by-page recovery for ${pass2Failed.reduce((s,t) => s + t.files.length, 0)} pages...`;

  await sleep(20000); // Longer reset before final pass

  const permanentlyFailed = [];

  for (const task of pass2Failed) {
    for (let i = 0; i < task.files.length; i++) {
      const singleFile = [task.files[i]];
      const pageOffset = task.offset + i;

      try {
        const entries = await extractWithSmartRetry(workDir, singleFile, pageOffset, docCtx, 2);
        allEntries.push(...entries);
        console.log(`[${jobId}] P3 ✓ page ${pageOffset + 1}`);
      } catch (err) {
        permanentlyFailed.push({ page: pageOffset + 1 });
        console.error(`[${jobId}] P3 ✗ page ${pageOffset + 1} — permanently failed`);
      }

      await sleep(5000); // 5s between single-page calls in Pass 3
    }
  }

  if (permanentlyFailed.length > 0) {
    console.warn(`[${jobId}] ⚠️ ${permanentlyFailed.length} pages permanently failed after all 3 passes`);
  } else {
    console.log(`[${jobId}] ✅ All pages recovered by Pass 3!`);
  }

  return { allEntries, failed: permanentlyFailed };
}

// ─── Smart retry with specific error handling ─────────────────────────────────
async function extractWithSmartRetry(workDir, chunkFiles, offset, docCtx, maxTries) {
  let lastErr;

  for (let attempt = 0; attempt <= maxTries; attempt++) {
    try {
      return await extractChunk(workDir, chunkFiles, offset, docCtx);
    } catch (err) {
      lastErr = err;
      const msg = (err.message || '').toLowerCase();

      if (attempt >= maxTries) break;

      // Rate limit (429) → must wait at least 65s
      if (msg.includes('429') || msg.includes('quota') || msg.includes('rate') || msg.includes('resource_exhausted')) {
        const waitMs = 65000 + (attempt * 15000);
        console.warn(`[Rate limit] Pass retry ${attempt+1}/${maxTries} — waiting ${waitMs/1000}s...`);
        await sleep(waitMs);
        continue;
      }

      // Service unavailable (503 / 500) → wait 20s
      if (msg.includes('503') || msg.includes('500') || msg.includes('unavailable')) {
        await sleep(20000 * (attempt + 1));
        continue;
      }

      // Permanent errors → don't retry
      if (msg.includes('safety') || msg.includes('400') || msg.includes('invalid')) {
        break;
      }

      // Generic errors → exponential backoff
      await sleep(3000 * Math.pow(2, attempt));
    }
  }
  throw lastErr;
}

// ─── Single chunk extraction ──────────────────────────────────────────────────
async function extractChunk(workDir, chunkFiles, offset, ctx) {
  const parts = [{ text: buildExtractionPrompt(chunkFiles.length, offset, ctx) }];

  chunkFiles.forEach((f, i) => {
    parts.push({ text: `=== PAGE ${offset + i + 1} ===` });
    parts.push({ inlineData: { mimeType: 'image/jpeg', data: b64(workDir, f) } });
  });

  const r = await ai.models.generateContent({
    model:    MODEL_NAME,
    contents: parts,
    config:   { responseMimeType: 'application/json', responseSchema: INDEX_SCHEMA }
  });

  const raw    = (r.text || '[]').replace(/```json|```/g, '').trim();
  const parsed = JSON.parse(raw);
  return Array.isArray(parsed) ? parsed : [];
}

// ─── Context analysis ─────────────────────────────────────────────────────────
async function analyzeContext(workDir, sampleFiles, totalPages) {
  const parts = [{ text: `You are an expert academic document analyst. Study these ${sampleFiles.length} sample pages from a ${totalPages}-page PDF.

Return ONLY valid JSON with NO extra text:
{
  "subject": "Academic subject (e.g. Zoology, Fluid Mechanics, Environmental Science)",
  "documentType": "Type (e.g. Assignment, Lab Report, Research Paper, Lecture Notes)",
  "theme": "Core theme in 5-10 words",
  "headingPattern": "Exactly how headings appear visually (e.g. Bold centered: Systematic Position of [Species Name])",
  "entryFormat": "How to write index entries (e.g. Keep Latin names italic, preserve capitalization)",
  "isHandwritten": false,
  "topicStructure": "SERIES | HIERARCHICAL | MIXED",
  "language": "Primary language (English, Bengali, Mixed)"
}` }];

  sampleFiles.forEach((f, i) => {
    parts.push({ text: `=== SAMPLE PAGE ${i + 1} ===` });
    parts.push({ inlineData: { mimeType: 'image/jpeg', data: b64(workDir, f) } });
  });

  try {
    const r = await ai.models.generateContent({
      model:    MODEL_NAME,
      contents: parts,
      config:   { responseMimeType: 'application/json', responseSchema: CONTEXT_SCHEMA }
    });
    const ctx = JSON.parse((r.text || '{}').replace(/```json|```/g, '').trim());
    if (!ctx.subject || !ctx.theme) throw new Error('Incomplete context');
    return ctx;
  } catch (err) {
    console.warn('Context fallback:', err.message);
    return {
      subject: 'Academic', documentType: 'Assignment', isHandwritten: false,
      theme: 'Academic document',
      headingPattern: 'Most prominent bold or centered text at top of new section',
      entryFormat: 'Main topic title as written',
      topicStructure: 'SERIES', language: 'English'
    };
  }
}

// ─── Extraction prompt ────────────────────────────────────────────────────────
function buildExtractionPrompt(pageCount, offset, ctx) {
  const handwrittenRules = `
HANDWRITTEN RULES:
  ✅ EXTRACT: Underlined text at start of topic
  ✅ EXTRACT: Text with ★ □ ☐ symbols
  ❌ SKIP: Taxonomy rows (Kingdom/Phylum/Class/Order/Family/Genus/Species)`;

  const typedRules = `
TYPED/PRINTED RULES:
  ✅ EXTRACT: Bold standalone line (not part of a sentence)
  ✅ EXTRACT: ALL-CAPS title line
  ✅ EXTRACT: Centered heading starting a new section
  ✅ EXTRACT: Numbered chapter heading (e.g. "Chapter 3: Heat Transfer")
  ❌ SKIP: Sub-headings that are not the primary topic title
  ❌ SKIP: Numbered list items (content, not headings)
  ❌ SKIP: Bold text inside a paragraph`;

  return `You are an expert academic index builder.

DOCUMENT CONTEXT:
  Subject      : ${ctx.subject}
  Type         : ${ctx.documentType}
  Theme        : "${ctx.theme}"
  Heading style: ${ctx.headingPattern}
  Entry format : ${ctx.entryFormat}
  Structure    : ${ctx.topicStructure}
  Language     : ${ctx.language}

${ctx.isHandwritten ? handwrittenRules : typedRules}

ALWAYS SKIP — NO EXCEPTIONS:
  ✗ Taxonomy: Kingdom / Phylum / Class / Order / Family / Genus / Species
  ✗ Figure captions: "Fig.", "Figure", "Diagram"
  ✗ Repeated page headers / footers
  ✗ Student name, roll, course code, institution name
  ✗ Dates (January 2025, 12/05/2024)
  ✗ "References", "Bibliography", "Contents"

RULES:
  1. Process ALL ${pageCount} pages — NEVER skip a page.
  2. Extract the SINGLE most prominent heading per page.
  3. COPY VERBATIM — exact spelling, capitalization, punctuation.
  4. NO heading on a page (pure diagram/blank/continuation)? → omit that page from output.
  5. Pages labeled "=== PAGE N ===" → use N as the page number.

Pages to analyze: ${offset + 1} to ${offset + pageCount}

Return ONLY a JSON array, no other text:
[{"page": N, "topic": "Exact heading text", "level": 0}]`;
}

// ─── Consolidate to ONE printed page ─────────────────────────────────────────
async function consolidate(allEntries, docCtx, max = ONE_PAGE_MAX) {
  const structure = (docCtx.topicStructure || 'SERIES').toUpperCase();
  const strategy  = structure.includes('SERIES')
    ? `SERIES: Select entries EVENLY distributed — cover beginning, middle, AND end. Do NOT cluster at the start.`
    : structure.includes('HIERARCHICAL')
    ? `HIERARCHICAL: Keep ONLY chapter-level entries. Remove all sub-headings.`
    : `MIXED: Keep one entry per major theme. Remove duplicates.`;

  const lastPage = allEntries[allEntries.length - 1]?.page || '?';
  const prompt = `Academic index editor. Printed page fits max ${max} entries.

Document: "${docCtx.theme}" (${docCtx.subject} ${docCtx.documentType})
Total available: ${allEntries.length} entries → Select exactly ${max}

STRATEGY: ${strategy}

RULES:
1. Output EXACTLY ${max} entries.
2. Keep original page number and topic text UNCHANGED.
3. Sort ascending by
