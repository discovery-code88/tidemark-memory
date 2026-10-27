//! Hybrid ranking: BM25 over record text (including date notes) and cosine
//! similarity of embeddings, fused by reciprocal rank.

use std::collections::HashMap;

const STOPWORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "for", "with", "from", "by", "about", "as",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "have", "has", "had", "i", "you",
    "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "what", "which", "who", "whom", "when", "where", "why", "how", "so", "if",
    "then", "than", "too", "very", "can", "could", "would", "should", "will", "just", "not", "no", "yes", "oh",
    "really", "there", "here", "all", "any", "some", "into", "out", "up", "down", "over", "also", "s", "t", "re",
    "ve", "ll", "d", "m", "im",
];

/// Lower-case word tokens without stop words, lightly stemmed.
/// Chinese, Japanese (kana) and Korean characters: written without spaces, so
/// they are indexed as overlapping character pairs.
fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3040..=0x30FF   // Hiragana, Katakana
        | 0x3400..=0x4DBF // CJK Extension A
        | 0x4E00..=0x9FFF // CJK Unified Ideographs
        | 0xAC00..=0xD7AF // Hangul syllables
        | 0xF900..=0xFAFF // CJK Compatibility Ideographs
        | 0x20000..=0x2FFFF)
}

/// Lower-case word tokens without stop words, lightly stemmed; runs of CJK
/// characters become overlapping two-character tokens (a single character
/// stays a token), so a partial phrase still matches.
pub fn tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for word in text.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()) {
        if !word.chars().any(is_cjk) {
            if !STOPWORDS.contains(&word) {
                out.push(stem(word));
            }
            continue;
        }
        // Split mixed runs into CJK and non-CJK stretches.
        let mut latin = String::new();
        let mut cjk: Vec<char> = Vec::new();
        let flush_cjk = |cjk: &mut Vec<char>, out: &mut Vec<String>| {
            if cjk.len() == 1 {
                out.push(cjk[0].to_string());
            }
            for pair in cjk.windows(2) {
                out.push(pair.iter().collect());
            }
            cjk.clear();
        };
        for c in word.chars() {
            if is_cjk(c) {
                if !latin.is_empty() {
                    if !STOPWORDS.contains(&latin.as_str()) {
                        out.push(stem(&latin));
                    }
                    latin.clear();
                }
                cjk.push(c);
            } else {
                flush_cjk(&mut cjk, &mut out);
                latin.push(c);
            }
        }
        flush_cjk(&mut cjk, &mut out);
        if !latin.is_empty() && !STOPWORDS.contains(&latin.as_str()) {
            out.push(stem(&latin));
        }
    }
    out
}

fn stem(word: &str) -> String {
    let w = word;
    for (suffix, min) in [("ies", 4), ("ing", 5), ("ed", 4), ("es", 4), ("s", 3)] {
        if w.len() > min && w.ends_with(suffix) && !w.ends_with("ss") {
            let base = &w[..w.len() - suffix.len()];
            // "boxes" -> "box", but "names" -> "name" (via the "s" rule).
            if suffix == "es" && !["s", "x", "z", "ch", "sh"].iter().any(|end| base.ends_with(end)) {
                continue;
            }
            return if suffix == "ies" { format!("{base}y") } else { base.to_string() };
        }
    }
    w.to_string()
}

pub struct Bm25<'a> {
    docs: Vec<&'a [String]>,
    df: HashMap<&'a str, usize>,
    avg_len: f32,
}

impl<'a> Bm25<'a> {
    pub fn new(docs: Vec<&'a [String]>) -> Self {
        let mut df = HashMap::new();
        for doc in &docs {
            let mut seen = std::collections::HashSet::new();
            for t in doc.iter() {
                if seen.insert(t.as_str()) {
                    *df.entry(t.as_str()).or_insert(0) += 1;
                }
            }
        }
        let avg_len = docs.iter().map(|d| d.len()).sum::<usize>() as f32 / docs.len().max(1) as f32;
        Bm25 { docs, df, avg_len: avg_len.max(1.0) }
    }

    pub fn scores(&self, query: &[String]) -> Vec<f32> {
        let n = self.docs.len() as f32;
        let (k1, b) = (1.2f32, 0.75f32);
        let mut q = query.to_vec();
        q.sort();
        q.dedup();
        self.docs
            .iter()
            .map(|doc| {
                let len = doc.len() as f32;
                q.iter()
                    .map(|term| {
                        let tf = doc.iter().filter(|t| *t == term).count() as f32;
                        if tf == 0.0 {
                            return 0.0;
                        }
                        let df = *self.df.get(term.as_str()).unwrap_or(&0) as f32;
                        let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                        idf * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * len / self.avg_len))
                    })
                    .sum()
            })
            .collect()
    }
}

/// Ranks of `scores` (0 = best); items with a zero score share the last rank.
fn ranks(scores: &[f32]) -> Vec<Option<usize>> {
    let mut order = (0..scores.len()).collect::<Vec<_>>();
    order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = vec![None; scores.len()];
    for (rank, i) in order.into_iter().enumerate() {
        if scores[i] > 0.0 {
            out[i] = Some(rank);
        }
    }
    out
}

/// Reciprocal-rank fusion of lexical and vector scores; returns indices and
/// fused scores, best first.
pub fn fuse(lexical: &[f32], vector: &[f32], lexical_weight: f32, k: f32) -> Vec<(usize, f32)> {
    let lr = ranks(lexical);
    let vr = ranks(&vector.iter().map(|s| s + 1.0).collect::<Vec<_>>());
    let mut fused = (0..lexical.len())
        .map(|i| {
            let l = lr[i].map_or(0.0, |r| lexical_weight / (k + r as f32 + 1.0));
            let v = vr[i].map_or(0.0, |r| (1.0 - lexical_weight) / (k + r as f32 + 1.0));
            (i, l + v)
        })
        .collect::<Vec<_>>();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fused
}

/// The known speakers (lower case) whose name appears as a word in `query`.
pub fn named_speakers<'a>(query: &str, speakers: impl Iterator<Item = &'a str>) -> Vec<String> {
    let words = query.to_lowercase().split(|c: char| !c.is_alphanumeric()).map(str::to_string).collect::<std::collections::HashSet<_>>();
    let mut out = Vec::new();
    for sp in speakers {
        if !out.iter().any(|o: &String| o == sp) && sp.split_whitespace().all(|part| words.contains(part)) {
            out.push(sp.to_string());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_lines_skip_headers_and_other_speakers() {
        let block = "[Friday 15 March 2024, 12:00]\nuser: Always use type hints.\nassistant: Sure.\nCaroline: hi";
        assert_eq!(user_lines_with(block, true), ["Always use type hints."]);
        assert_eq!(user_lines_with(block, false), ["Always use type hints."]);
        assert_eq!(user_lines_with("[Friday 15 March 2024, 12:00] user: one long part", true), ["one long part"]);
        assert!(user_lines_with("Caroline: user: not a role", true).is_empty());
    }

    #[test]
    fn multiline_user_messages_keep_all_their_lines() {
        let block = "[Friday 15 March 2024, 12:00]\nuser: Here are my rules:\n1. Always use type hints\n2. Never use tabs\nassistant: Noted.\nmore reply text\nuser: thanks";
        assert_eq!(user_lines_with(block, true), ["Here are my rules:", "1. Always use type hints", "2. Never use tabs", "thanks"]);
        assert_eq!(user_lines_with(block, false), ["Here are my rules:", "thanks"], "default: first lines only, as evaluated");
    }

    #[test]
    fn standing_instructions_are_detected() {
        assert!(is_standing_instruction("Always format code snippets with syntax highlighting when I ask."));
        assert!(is_standing_instruction("Thanks. Never use tabs in my code."));
        assert!(is_standing_instruction("I'm learning, and I prefer step-by-step explanations."));
        assert!(is_standing_instruction("From now on, answer in French."));
        assert!(!is_standing_instruction("I always forget my keys."));
        assert!(!is_standing_instruction("I have never used Flask before."));
        assert!(!is_standing_instruction("I want to make sure the app is secure."));
    }

    #[test]
    fn tokens_drop_stop_words_and_stem() {
        assert_eq!(tokens("I went camping with the kids!"), ["went", "camp", "kid"]);
        assert_eq!(tokens("Her puppies' names"), ["puppy", "name"]);
    }

    #[test]
    fn cjk_text_is_indexed_as_character_pairs() {
        assert_eq!(tokens("独角兽"), ["独角", "角兽"]);
        assert_eq!(tokens("我"), ["我"]);
        assert_eq!(tokens("東京タワーへ行った"), ["東京", "京タ", "タワ", "ワー", "ーへ", "へ行", "行っ", "った"]);
        assert_eq!(tokens("iPhone买了"), ["iphone", "买了"]);
        assert_eq!(tokens("I went camping"), ["went", "camp"], "English is unchanged");
    }

    #[test]
    fn a_partial_cjk_phrase_finds_its_sentence() {
        let docs = [tokens("我们周末去了海边看日落"), tokens("他昨天买了一辆新自行车"), tokens("周末我在家看书")];
        let bm = Bm25::new(docs.iter().map(|d| d.as_slice()).collect());
        let s = bm.scores(&tokens("海边日落"));
        assert!(s[0] > s[1] && s[0] > s[2], "{s:?}");
    }

    #[test]
    fn bm25_prefers_the_document_with_rare_query_terms() {
        let docs = [tokens("I love painting sunsets"), tokens("We went camping in the mountains"), tokens("Painting is fun")];
        let bm = Bm25::new(docs.iter().map(|d| d.as_slice()).collect());
        let s = bm.scores(&tokens("Where did they go camping?"));
        assert!(s[1] > s[0] && s[1] > s[2]);
    }

    #[test]
    fn speakers_named_in_the_query_are_found() {
        let named = named_speakers("What activities does Melanie partake in?", ["caroline", "melanie", "melanie"].into_iter());
        assert_eq!(named, ["melanie"]);
    }

    #[test]
    fn fusion_rewards_agreement() {
        let fused = fuse(&[3.0, 2.0, 0.0], &[0.1, 0.9, 0.2], 0.5, 60.0);
        assert_eq!(fused[0].0, 1);
    }
}

/// The user's own lines of a record, without the date header: lines starting
/// with "user:", and with TIDEMARK_USER_LINES=continued also the lines that
/// follow them up to the next role (multiline messages). The default keeps
/// only the "user:" lines, as in the evaluated configuration.
pub fn user_lines(content: &str) -> Vec<&str> {
    static CONTINUED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    user_lines_with(content, *CONTINUED.get_or_init(|| std::env::var("TIDEMARK_USER_LINES").is_ok_and(|v| v.trim().eq_ignore_ascii_case("continued"))))
}

pub fn user_lines_with(content: &str, continued: bool) -> Vec<&str> {
    let mut out = Vec::new();
    let mut in_user = false;
    for line in content.lines() {
        let line = line.trim();
        let line = match line.strip_prefix('[').and_then(|rest| rest.split_once("] ")) {
            Some((_, rest)) => rest.trim_start(),
            None if line.starts_with('[') && line.ends_with(']') => {
                in_user = false; // a header line on its own starts a new record
                continue;
            }
            None => line,
        };
        let role = ["user:", "assistant:", "system:", "tool:"].into_iter().find(|r| line.get(..r.len()).is_some_and(|h| h.eq_ignore_ascii_case(r)));
        match role {
            Some(r) => {
                in_user = r == "user:";
                let rest = line[r.len()..].trim();
                if in_user && !rest.is_empty() {
                    out.push(rest);
                }
            }
            None if continued && in_user && !line.is_empty() => out.push(line),
            None => {}
        }
    }
    out
}

/// Whether a user line states a standing instruction or preference ("Always
/// include …", "Never …", "From now on …", "I prefer …").
pub fn is_standing_instruction(line: &str) -> bool {
    static CUE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    CUE.get_or_init(|| {
        regex::Regex::new(
            r"(?i)(?:^|[.!?]\s+|\bplease\s+)(?:always|never|from now on|going forward|make sure)\b|\bi(?: would|'d|’d)? (?:really |strongly )?prefer\b|\bkeep in mind\b|\bplease remember\b",
        )
        .unwrap()
    })
    .is_match(line)
}
