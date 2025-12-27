import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, statSync, unlinkSync } from 'node:fs';
import { join, dirname, delimiter } from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const workspaceRoot = join(__dirname, '..', '..');
const outDir = join(__dirname, '..', 'src', 'lib', 'pixo-wasm');
const wasmPath = join(workspaceRoot, 'target', 'wasm32-unknown-unknown', 'release', 'pixo.wasm');

function formatError(message) {
	console.error(`\n[wasm:build] ${message}`);
	process.exit(1);
}

function findOnPath(bin) {
	const paths = (process.env.PATH || '').split(delimiter);
	for (const p of paths) {
		const candidate = join(p, bin);
		if (existsSync(candidate)) return candidate;
	}
	return null;
}

function findCachedWasmBindgen() {
	const cacheRoot = join(os.homedir(), '.cache', '.wasm-pack');
	if (!existsSync(cacheRoot)) return null;
	const entries = readdirSync(cacheRoot, { withFileTypes: true });
	for (const entry of entries) {
		if (entry.isDirectory() && entry.name.startsWith('wasm-bindgen-')) {
			const candidate = join(cacheRoot, entry.name, 'wasm-bindgen');
			if (existsSync(candidate)) return candidate;
		}
	}
	return null;
}

function ensureOutDir() {
	if (!existsSync(outDir)) {
		mkdirSync(outDir, { recursive: true });
	}
}

function findWasmBindgenBinary() {
	if (process.env.WASM_BINDGEN && existsSync(process.env.WASM_BINDGEN)) {
		return process.env.WASM_BINDGEN;
	}
	const fromPath = findOnPath('wasm-bindgen');
	if (fromPath) return fromPath;
	const cached = findCachedWasmBindgen();
	if (cached) return cached;
	formatError(
		'wasm-bindgen binary not found. Install wasm-pack (downloads wasm-bindgen) or install wasm-bindgen-cli and set WASM_BINDGEN to its path.'
	);
	return null;
}

function removeGitIgnore() {
	const gitignore = join(outDir, '.gitignore');
	if (existsSync(gitignore)) {
		unlinkSync(gitignore);
	}
}

function findWasmOpt() {
	const fromPath = findOnPath('wasm-opt');
	if (fromPath) return fromPath;
	console.warn('[wasm:build] wasm-opt not found on PATH; skipping optimization step.');
	return null;
}

// Find the correct cargo/rustc binaries. Prefer rustup-managed versions over Homebrew
// because Homebrew Rust doesn't have wasm32-unknown-unknown target installed.
function findRustupBinaries() {
	const home = os.homedir();
	const rustupCargo = join(home, '.cargo', 'bin', 'cargo');
	const rustupRustc = join(home, '.cargo', 'bin', 'rustc');

	// Check if rustup versions exist
	if (existsSync(rustupCargo) && existsSync(rustupRustc)) {
		return { cargo: rustupCargo, rustc: rustupRustc };
	}

	// Fall back to PATH (will fail if only Homebrew Rust is installed)
	return { cargo: 'cargo', rustc: null };
}

try {
	const { cargo, rustc } = findRustupBinaries();

	console.log('[wasm:build] Building Rust crate (wasm32-unknown-unknown, release)...');

	// Set RUSTC to rustup's rustc to avoid "can't find crate for core" error
	// when Homebrew Rust is also installed
	const env = { ...process.env };
	if (rustc) {
		env.RUSTC = rustc;
	}

	execFileSync(
		cargo,
		[
			'build',
			'--target',
			'wasm32-unknown-unknown',
			'--release',
			'--no-default-features',
			'--features',
			'wasm,simd',
		],
		{ cwd: workspaceRoot, stdio: 'inherit', env }
	);

	const bindgen = findWasmBindgenBinary();
	ensureOutDir();

	console.log('[wasm:build] Running wasm-bindgen...');
	execFileSync(
		bindgen,
		['--target', 'web', '--out-dir', outDir, '--out-name', 'pixo', wasmPath],
		{ cwd: workspaceRoot, stdio: 'inherit' }
	);

	removeGitIgnore();

	if (!existsSync(wasmPath) || !existsSync(join(outDir, 'pixo_bg.wasm'))) {
		formatError('Build finished, but output files are missing.');
	}

	const wasmFile = join(outDir, 'pixo_bg.wasm');
	const wasmOpt = findWasmOpt();
	if (wasmOpt) {
		const pre = statSync(wasmFile).size;
		console.log('[wasm:build] Running wasm-opt -Oz (strip debug/producers/target-features)...');
		execFileSync(
			wasmOpt,
			[
				'-Oz',
				'--strip-debug',
				'--strip-dwarf',
				'--strip-producers',
				'--strip-target-features',
				'--enable-bulk-memory',
				'--enable-sign-ext',
				'--enable-nontrapping-float-to-int',
				'-o',
				wasmFile,
				wasmFile,
			],
			{ cwd: workspaceRoot, stdio: 'inherit' }
		);
		const post = statSync(wasmFile).size;
		console.log(`[wasm:build] wasm-opt reduced size ${pre} -> ${post} bytes.`);
	}

	const wasmSize = statSync(wasmFile).size;
	console.log(`[wasm:build] Done. Emitted ${wasmSize} bytes of wasm to ${outDir}.`);
} catch (err) {
	formatError(err instanceof Error ? err.message : String(err));
}
