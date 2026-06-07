//! proxy_verify — Rust ground-truth driver for the proxy RL verifier equivalence test.
//!
//! Uses the REAL challenge simulator semantics: `circuit.rs` and `sim.rs` are byte-identical
//! vendored copies of `ecdsafail-challenge/src/{circuit,sim}.rs`. We parse op streams with the
//! real `Op::from_text`, size the sim with the real `analyze_ops`, and run the real
//! `Simulator::apply_iter` over enumerated basis inputs, packing 64 shots per u64 exactly like
//! the harness does.
//!
//! Protocol (line-delimited, for batch throughput in the equivalence test):
//!   * Each stdin line is one request. Two accepted forms:
//!       1. JSON object: {"opstream": "<text with \\n>", "width": N}
//!       2. Two tab-separated fields: <width>\t<opstream-with-\n-escaped-as-\\n>
//!     Both produce the same result line on stdout.
//!   * Alternatively (single-shot mode): argv[1] = path to op-stream text file, argv[2] = width.
//!
//! For each request we enumerate all 2^width basis input states (ancillas are part of the
//! enumerated state — full state space), in batches of up to 64 input states packed into the
//! shot-lanes, matching the Python reference harness batch boundaries EXACTLY. We print one
//! JSON line:
//!   {"toffoli": <total executed Toffoli over all batches>,
//!    "peak_width": <analyze_ops num_qubits>,
//!    "final_states": [<full-width output state per basis input, index 0..2^width>]}

// Vendored byte-identical copies of the challenge's circuit.rs/sim.rs. They expose more API
// than this driver uses (set_register/get_register/Circuit::from_text/etc.); allow dead code so
// the vendored sources stay verbatim without warning noise.
#[allow(dead_code)]
mod circuit;
#[allow(dead_code)]
mod sim;

use circuit::{analyze_ops, Op};
use sim::Simulator;
use sha3::{
    digest::{ExtendableOutput, Update},
    Shake256,
};
use std::io::{self, BufRead, Read, Write};

/// Parse an op-stream text into Vec<Op> using the REAL grammar (`Op::from_text`).
/// Lines that are blank/comment produce None and are skipped, exactly like Circuit::from_text.
fn parse_ops(text: &str) -> Vec<Op> {
    let mut ops = Vec::new();
    for line in text.lines() {
        if let Some(op) = Op::from_text(line) {
            ops.push(op);
        }
    }
    ops
}

/// Full result of enumerating a circuit over all 2^width basis states.
struct CircuitResult {
    total_toffoli: u64,
    peak_width: u64,
    final_states: Vec<u64>, // full qubit state per basis input
    phase_bits: Vec<u64>,   // phase lane per basis input (0/1)
    bit_states: Vec<u64>,   // final classical-bit register value per basis input
}

/// Run the real simulator over all 2^width basis states (full state enumeration). 64 inputs per
/// batch, lanes 0..batch loaded with inputs idx..idx+batch; this matches the Python harness batch
/// boundaries. Classical bits start at 0 (clear_for_shot) and are read back after the run.
fn run_circuit(text: &str, width: u32) -> CircuitResult {
    let ops = parse_ops(text);
    let (nq, nb, _nr, _regs) = analyze_ops(ops.iter());

    let num_qubits = (width as u64).max(nq) as usize;
    let num_bits = nb as usize;

    let n_states: u64 = 1u64 << width;
    let mut final_states = vec![0u64; n_states as usize];
    let mut phase_bits = vec![0u64; n_states as usize];
    let mut bit_states = vec![0u64; n_states as usize];
    let mut total_toffoli: u64 = 0;

    let mut idx: u64 = 0;
    while idx < n_states {
        let batch = std::cmp::min(64u64, n_states - idx);

        // Fresh XOF per batch (matches Python harness, which constructs a new XOF per batch).
        let mut xof = {
            let mut h = Shake256::default();
            h.update(b"proxy-seed");
            h.finalize_xof()
        };
        let mut s = Simulator::new(num_qubits, num_bits, &mut xof);
        s.clear_for_shot();

        // load basis states into shot lanes
        for lane in 0..batch {
            let inp = idx + lane;
            for q in 0..width {
                if (inp >> q) & 1 == 1 {
                    s.qubits[q as usize] |= 1u64 << lane;
                }
            }
        }

        let tof_before = s.stats.toffoli_gates;
        s.apply_iter(ops.iter());
        total_toffoli += s.stats.toffoli_gates - tof_before;

        for lane in 0..batch {
            let inp = idx + lane;
            let mut st: u64 = 0;
            for q in 0..width {
                if (s.qubits[q as usize] >> lane) & 1 == 1 {
                    st |= 1u64 << q;
                }
            }
            final_states[inp as usize] = st;
            phase_bits[inp as usize] = (s.phase >> lane) & 1;
            let mut bst: u64 = 0;
            for b in 0..num_bits {
                if (s.bits[b] >> lane) & 1 == 1 {
                    bst |= 1u64 << b;
                }
            }
            bit_states[inp as usize] = bst;
        }

        idx += batch;
    }

    CircuitResult {
        total_toffoli,
        peak_width: nq,
        final_states,
        phase_bits,
        bit_states,
    }
}

/// Minimal JSON string unescaper for the subset we emit (\\n, \\t, \\\\, \\", \\/).
fn json_unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('\\') => out.push('\\'),
                Some('"') => out.push('"'),
                Some('/') => out.push('/'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Extract the string value of a top-level JSON key "key":"...". Returns the raw (still-escaped)
/// inner string. Hand-rolled to avoid a serde_json dependency; input is produced by our own
/// test harness so the format is controlled.
fn json_get_str<'a>(json: &'a str, key: &str) -> Option<String> {
    let pat = format!("\"{}\"", key);
    let kpos = json.find(&pat)?;
    let after = &json[kpos + pat.len()..];
    // find the colon
    let colon = after.find(':')?;
    let mut rest = after[colon + 1..].trim_start();
    if !rest.starts_with('"') {
        return None;
    }
    rest = &rest[1..];
    // scan to the closing unescaped quote
    let bytes = rest.as_bytes();
    let mut i = 0;
    let mut end = None;
    while i < bytes.len() {
        if bytes[i] == b'\\' {
            i += 2;
            continue;
        }
        if bytes[i] == b'"' {
            end = Some(i);
            break;
        }
        i += 1;
    }
    let end = end?;
    Some(rest[..end].to_string())
}

fn json_get_u64(json: &str, key: &str) -> Option<u64> {
    let pat = format!("\"{}\"", key);
    let kpos = json.find(&pat)?;
    let after = &json[kpos + pat.len()..];
    let colon = after.find(':')?;
    let rest = after[colon + 1..].trim_start();
    let mut num = String::new();
    for c in rest.chars() {
        if c.is_ascii_digit() {
            num.push(c);
        } else {
            break;
        }
    }
    num.parse().ok()
}

fn push_u64_array(s: &mut String, arr: &[u64]) {
    s.push('[');
    for (i, v) in arr.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&v.to_string());
    }
    s.push(']');
}

fn emit_result(out: &mut impl Write, r: &CircuitResult) {
    let mut s = String::new();
    s.push_str("{\"toffoli\":");
    s.push_str(&r.total_toffoli.to_string());
    s.push_str(",\"peak_width\":");
    s.push_str(&r.peak_width.to_string());
    s.push_str(",\"final_states\":");
    push_u64_array(&mut s, &r.final_states);
    s.push_str(",\"phase_bits\":");
    push_u64_array(&mut s, &r.phase_bits);
    s.push_str(",\"bit_states\":");
    push_u64_array(&mut s, &r.bit_states);
    s.push('}');
    writeln!(out, "{}", s).unwrap();
}

/// Run with catch_unwind so a malformed op stream (panic in from_text/validate) yields an
/// {"error": ...} line rather than killing the process — mirrors the harness's hardened loader.
fn handle_request(out: &mut impl Write, text: &str, width: u32) {
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_circuit(text, width)));
    match res {
        Ok(r) => emit_result(out, &r),
        Err(_) => {
            writeln!(out, "{{\"error\":\"panic\"}}").unwrap();
        }
    }
}

fn main() {
    // Silence panic messages (we catch them) so stderr stays clean during the equivalence test.
    std::panic::set_hook(Box::new(|_| {}));

    let args: Vec<String> = std::env::args().collect();

    // Single-shot file mode: argv[1] = path, argv[2] = width.
    if args.len() >= 3 {
        let path = &args[1];
        let width: u32 = args[2].parse().expect("width must be an integer");
        let mut text = String::new();
        std::fs::File::open(path)
            .expect("open op-stream file")
            .read_to_string(&mut text)
            .expect("read op-stream file");
        let stdout = io::stdout();
        let mut lock = stdout.lock();
        handle_request(&mut lock, &text, width);
        return;
    }

    // Line-delimited request mode (default): one request per stdin line.
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }

        let (text, width) = if line.trim_start().starts_with('{') {
            // JSON form
            let width = json_get_u64(&line, "width").unwrap_or(0) as u32;
            let raw = json_get_str(&line, "opstream").unwrap_or_default();
            (json_unescape(&raw), width)
        } else {
            // TSV form: <width>\t<opstream with \n escaped>
            let mut parts = line.splitn(2, '\t');
            let w = parts.next().unwrap_or("0");
            let body = parts.next().unwrap_or("");
            let width: u32 = w.trim().parse().unwrap_or(0);
            (json_unescape(body), width)
        };

        handle_request(&mut out, &text, width);
    }
    out.flush().unwrap();
}
