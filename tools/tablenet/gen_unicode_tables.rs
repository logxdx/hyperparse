// Emits C++ range tables of Rust's char classes (the reference's semantics), for
// src/tablenet/unicode_tables.inc.
fn ranges(f: impl Fn(char) -> bool) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = Vec::new();
    for u in 0..=0x10FFFFu32 {
        if let Some(c) = char::from_u32(u) {
            if f(c) {
                match out.last_mut() {
                    Some(r) if r.1 + 1 == u => r.1 = u,
                    _ => out.push((u, u)),
                }
            }
        }
    }
    out
}

fn main() {
    println!("// Generated from Rust's char methods (is_alphabetic, is_uppercase, is_lowercase,");
    println!("// is_numeric, is_whitespace): inclusive code point ranges, sorted. Do not edit by hand.");
    let sets: [(&str, fn(char) -> bool); 5] = [
        ("kAlphabetic", |c| c.is_alphabetic()),
        ("kUppercase", |c| c.is_uppercase()),
        ("kLowercase", |c| c.is_lowercase()),
        ("kNumeric", |c| c.is_numeric()),
        ("kWhitespace", |c| c.is_whitespace()),
    ];
    for (name, f) in sets {
        let r = ranges(f);
        println!("inline constexpr Range {name}[] = {{");
        for chunk in r.chunks(6) {
            let s: Vec<String> = chunk.iter().map(|(a, b)| format!("{{0x{a:X}, 0x{b:X}}}")).collect();
            println!("    {},", s.join(", "));
        }
        println!("}};");
    }
}
