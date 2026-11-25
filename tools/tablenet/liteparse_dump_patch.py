"""Add LP_TN_NOPROJ (drop projection signals) and LP_TN_DUMP=<dir> (per-page dump) to the
reference liteparse table model. Run from crates/liteparse/src."""
p = 'table_model/mod.rs'; s = open(p).read()
def rep(a, b):
    global s
    assert a in s, a[:80]
    s = s.replace(a, b, 1)
rep('''pub const MAX_UNITS: usize = 1500;''', '''pub const MAX_UNITS: usize = 1500;

thread_local! {
    static DUMP_PAGE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
/// Debug: page number for `LP_TN_DUMP` file names.
pub fn set_dump_page(p: usize) {
    DUMP_PAGE.with(|c| c.set(p));
}''')
rep('''        let (us, proj) = ocr_units(input, s);
        (us.into(), proj)''', '''        let (us, mut proj) = ocr_units(input, s);
        if std::env::var_os("LP_TN_NOPROJ").is_some() {
            proj = vec![None; proj.len()];
        }
        (us.into(), proj)''')
rep('''        let proj = match sig {
            Some(s) => join_signals(input, s),
            None => vec![None; input.units.len()],
        };''', '''        let proj = match sig {
            Some(s) if std::env::var_os("LP_TN_NOPROJ").is_none() => join_signals(input, s),
            _ => vec![None; input.units.len()],
        };''')
rep('''    let f32x4 = |b: [f64; 4]| [b[0] as f32, b[1] as f32, b[2] as f32, b[3] as f32];
    let fu: Vec<features::FUnit> = units''', '''    let f32x4 = |b: [f64; 4]| [b[0] as f32, b[1] as f32, b[2] as f32, b[3] as f32];
    let proj_dump: Vec<Option<features::Proj>> = proj.clone();
    let fu: Vec<features::FUnit> = units''')
rep('''    let mut logits = net.forward(&feats, &pairf, n, img.as_ref());
    let minor: Vec<bool> = units.iter().map(|u| u.minor).collect();
    mask_minor(&mut logits, &minor);''', '''    let mut logits = net.forward(&feats, &pairf, n, img.as_ref());
    let raw_logits = (logits.node.clone(), logits.pair.clone(), logits.cls.clone());
    let minor: Vec<bool> = units.iter().map(|u| u.minor).collect();
    mask_minor(&mut logits, &minor);
    let mut dump_tables: Vec<serde_json::Value> = Vec::new();''')
rep('''        out.push(ModelTable { bbox, grid });
    }''', '''        if std::env::var_os("LP_TN_DUMP").is_some() {
            dump_tables.push(serde_json::json!({"units": g, "html": grid.to_html(),
                "n_rows": grid.n_rows, "n_cols": grid.n_cols, "head_rows": grid.head_rows,
                "bbox": [bbox.x, bbox.y, bbox.width, bbox.height]}));
        }
        out.push(ModelTable { bbox, grid });
    }
    if let Some(dir) = std::env::var_os("LP_TN_DUMP") {
        let page = DUMP_PAGE.with(|c| c.get());
        let dir = std::path::PathBuf::from(dir);
        let small = n <= 600;
        let pj: Vec<serde_json::Value> = proj_dump.iter().map(|p| match p {
            Some(p) => serde_json::json!({"flow": p.flow, "lc": p.lc, "rc": p.rc, "cc": p.cc,
                "snap": p.snap.to_string(), "block": p.block, "leaf": p.leaf,
                "feat_text": p.feat_text}),
            None => serde_json::Value::Null,
        }).collect();
        let us: Vec<serde_json::Value> = units.iter().map(|u| serde_json::json!({
            "bx": u.bx, "pbox": u.pbox, "text": u.text, "h": u.h, "size": u.size,
            "bold": u.bold, "q": u.q, "minor": u.minor,
            "words": u.words.iter().map(|w| format!("{:?}", w)).collect::<Vec<_>>()})).collect();
        let j = serde_json::json!({
            "page": page, "wh": [input.wh.0, input.wh.1], "ocr": input.ocr, "n": n,
            "units": us, "proj": pj, "feats": feats,
            "pairf": if small { serde_json::json!(pairf) } else { serde_json::Value::Null },
            "rules": rules.map(|r| serde_json::json!({"h": r.h, "v": r.v})),
            "img": page_img.map(|im| serde_json::json!({"w": im.w, "h": im.h, "k": im.k})),
            "img_valid": img.as_ref().map(|i| [i.valid.0, i.valid.1]),
            "img_boxes": img.as_ref().map(|i| i.boxes.clone()),
            "node": raw_logits.0,
            "pair": if small { serde_json::json!(raw_logits.1) } else { serde_json::Value::Null },
            "cls": raw_logits.2,
            "tables": dump_tables,
        });
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(dir.join(format!("p{page:04}.json")), j.to_string());
        if let Some(im) = page_img {
            let _ = std::fs::write(dir.join(format!("p{page:04}_img.u8")), &im.px);
        }
    }''')
open(p, 'w').write(s)
p = 'parser.rs'; s = open(p).read()
rep('''                let run = |(page_number, input): &(usize, crate::table_model::PageInput)| {
                    crate::table_model::run_page(''', '''                let run = |(page_number, input): &(usize, crate::table_model::PageInput)| {
                    crate::table_model::set_dump_page(*page_number);
                    crate::table_model::run_page(''')
open(p, 'w').write(s)
print("patched")
