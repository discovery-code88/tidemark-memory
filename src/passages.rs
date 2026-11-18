//! The best-matching passage of a long record (`TIDEMARK_PASSAGES=on`, per search `passages: true`): within one
//! message of the record, the run of consecutive sentences (at most `TIDEMARK_PASSAGE_MAX_CHARS`, default 400) that
//! covers the query's words best, rarer words counting more (the user's BM25 statistics). Returned next to the
//! record as `passage: {text, speaker, start, end, matched_terms}`; `start`/`end` are character offsets (Unicode code
//! points) into the record's `content`, which stays complete. Records shorter than `TIDEMARK_PASSAGE_MIN_WORDS`
//! (default 120) get none: they are their own passage. Nothing is returned when no query word occurs in the record
//! (a match by meaning alone); the record is still returned as usual.
//!
//! The choice is lexical: no embedding calls, so it costs nothing with paid providers, and it can miss a passage that
//! matches only by meaning or leave out a qualification stated elsewhere. The record stays authoritative; the
//! passage is a reading aid.

use crate::search;

#[derive(Debug, Clone, PartialEq)]
pub struct Passage {
    pub text: String,
    pub speaker: Option<String>,
    pub start: usize,
    pub end: usize,
    pub matched: Vec<String>,
}

impl Passage {
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({"text": self.text, "speaker": self.speaker, "start": self.start, "end": self.end, "matched_terms": self.matched})
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub max_chars: usize,
    pub min_words: usize,
}

/// Byte spans of the sentences of one message body (`from..to` in `content`), split after . ! ? followed by
/// whitespace, after the CJK marks 。！？, and at line breaks.
fn sentences(content: &str, from: usize, to: usize) -> Vec<(usize, usize)> {
    let body = &content[from..to];
    let mut out = Vec::new();
    let mut start = 0usize;
    let chars = body.char_indices().collect::<Vec<_>>();
    for (k, &(i, c)) in chars.iter().enumerate() {
        let next = chars.get(k + 1).map(|x| x.1);
        // . ! ? end a sentence before whitespace; the CJK marks 。！？ end one wherever they stand.
        let end_here = c == '\n' || matches!(c, '。' | '！' | '？') || (matches!(c, '.' | '!' | '?') && next.is_none_or(char::is_whitespace));
        if end_here {
            let stop = i + c.len_utf8();
            if body[start..stop].trim().len() > 1 {
                out.push((from + start, from + stop));
            }
            start = stop;
        }
    }
    if body[start..].trim().len() > 1 {
        out.push((from + start, to));
    }
    // Trim surrounding whitespace off each span.
    out.into_iter()
        .map(|(a, b)| {
            let s = &content[a..b];
            let lead = s.len() - s.trim_start().len();
            let trail = s.len() - s.trim_end().len();
            (a + lead, b - trail)
        })
        .filter(|(a, b)| b > a)
        .collect()
}

/// The messages of a record: (speaker, body byte span). A line starting with "name: " (after an optional
/// "[date …] " header) opens a message; other lines continue the current one.
fn messages(content: &str) -> Vec<(Option<String>, usize, usize)> {
    let mut out: Vec<(Option<String>, usize, usize)> = Vec::new();
    let mut offset = 0usize;
    for line in content.split_inclusive('\n') {
        let mut body_start = offset;
        let mut rest = line;
        if rest.starts_with('[') {
            if let Some(close) = rest.find("] ").or_else(|| rest.trim_end().ends_with(']').then(|| rest.trim_end().len() - 1)) {
                let skip = (close + 2).min(rest.len());
                body_start += skip;
                rest = &rest[skip..];
            }
        }
        let speaker = rest.find(": ").filter(|&p| p > 0 && p <= 40 && !rest[..p].contains(['.', '!', '?', '\n'])).map(|p| (p, rest[..p].trim().to_string()));
        let line_end = offset + line.len();
        match speaker {
            Some((p, name)) => out.push((Some(name), body_start + p + 2, line_end)),
            None => match out.last_mut() {
                Some(last) => last.2 = line_end,
                None if rest.trim().is_empty() => {}
                None => out.push((None, body_start, line_end)),
            },
        }
        offset = line_end;
    }
    out
}

/// Byte span of at most `max_chars` characters inside `a..b`, starting at the word boundary about half the limit
/// before the first word whose token is in `covered`.
fn window(content: &str, a: usize, b: usize, covered: &std::collections::HashSet<String>, max_chars: usize) -> (usize, usize) {
    let s = &content[a..b];
    let mut first = 0usize;
    let mut pos = 0usize;
    for word in s.split_inclusive(char::is_whitespace) {
        if search::tokens(word).iter().any(|t| covered.iter().any(|c| key(c) == key(t))) {
            first = pos;
            break;
        }
        pos += word.len();
    }
    let back = max_chars / 2;
    let chars_before = s[..first].chars().count();
    let mut start = s.char_indices().nth(chars_before.saturating_sub(back)).map_or(0, |(k, _)| k);
    if start > 0 {
        // Move to the next word boundary so the passage does not open mid-word.
        start = s[start..].find(char::is_whitespace).map_or(start, |w| start + w + 1).min(first);
    }
    let end = s[start..].char_indices().nth(max_chars).map_or(s.len(), |(k, _)| start + k);
    (a + start, a + end)
}

/// A token with a final "e" removed and a doubled final consonant made single (see `best`).
fn key(token: &str) -> String {
    let mut k = token.strip_suffix('e').filter(|r| r.chars().count() >= 3).unwrap_or(token).to_string();
    let cs = k.chars().collect::<Vec<_>>();
    if cs.len() >= 4 && cs[cs.len() - 1] == cs[cs.len() - 2] && cs[cs.len() - 1].is_alphabetic() && !"aeiou".contains(cs[cs.len() - 1]) {
        k.pop();
    }
    k
}

/// Length in words for `min_words`: whitespace-separated words, with each CJK character counted as one (CJK text
/// has no spaces between words).
fn words(content: &str) -> usize {
    content.split_whitespace().map(|w| { let cjk = w.chars().filter(|c| search::is_cjk(*c)).count(); if cjk > 0 { cjk } else { 1 } }).sum()
}

fn char_offset(content: &str, byte: usize) -> usize {
    content[..byte].chars().count()
}

/// The best passage of `content` for the query, or None (short record, or no query word in it).
pub fn best(content: &str, query_tokens: &[String], idf: impl Fn(&str) -> f32, opts: Options) -> Option<Passage> {
    if words(content) < opts.min_words || query_tokens.is_empty() {
        return None;
    }
    let wanted = query_tokens.iter().cloned().collect::<std::collections::HashSet<_>>();
    // A text word counts for a query word when their tokens are equal after undoing the two ways the service's light
    // stemmer splits one word: a dropped final "e" (settle → `settle`, settled → `settl`) and a doubled final
    // consonant (cap → `cap`, capped → `capp`). Used only here, not for ranking.
    let keyed = wanted.iter().map(|w| (key(w), w.clone())).collect::<std::collections::HashMap<_, _>>();
    let hit = |t: &String| -> Option<String> { if wanted.contains(t) { Some(t.clone()) } else { keyed.get(&key(t)).cloned() } };
    let mut best: Option<(f32, usize, Passage)> = None;
    for (speaker, from, to) in messages(content) {
        let spans = sentences(content, from, to);
        let toks = spans.iter().map(|&(a, b)| search::tokens(&content[a..b]).iter().filter_map(&hit).collect::<std::collections::HashSet<_>>()).collect::<Vec<_>>();
        for i in 0..spans.len() {
            let mut covered = std::collections::HashSet::new();
            for j in i..spans.len() {
                let (a, b) = (spans[i].0, spans[j].1);
                if j > i && content[a..b].chars().count() > opts.max_chars {
                    break;
                }
                covered.extend(toks[j].iter().cloned());
                if covered.is_empty() {
                    continue;
                }
                let score = covered.iter().map(|t| idf(t)).sum::<f32>();
                let len = content[a..b].chars().count();
                let better = match &best {
                    None => true,
                    Some((s, l, _)) => score > *s + 1e-6 || ((score - *s).abs() <= 1e-6 && len < *l),
                };
                if better {
                    let (mut a, mut b) = (a, b);
                    if len > opts.max_chars {
                        // One sentence longer than the limit: a window of the limit's size around its first matching
                        // word, starting at a word boundary.
                        (a, b) = window(content, a, b, &covered, opts.max_chars);
                    }
                    let text = content[a..b].to_string();
                    let end = b;
                    let mut matched = covered.iter().cloned().collect::<Vec<_>>();
                    matched.sort();
                    best = Some((score, len, Passage { text, speaker: speaker.clone(), start: char_offset(content, a), end: char_offset(content, end), matched }));
                }
            }
        }
    }
    best.map(|(_, _, p)| p)
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPTS: Options = Options { max_chars: 400, min_words: 20 };

    fn q(text: &str) -> Vec<String> {
        search::tokens(text)
    }

    #[test]
    fn the_matching_paragraph_of_a_long_mixed_record_is_found_with_its_speaker_and_offsets() {
        let content = "[Tuesday 17 November 2026, 05:32] \nuser: Agreed. Close selection-first as a negative development result and keep it off. \
The audit justifies stopping.\nuser: That closes this line of work. Use the guide when failures arise. Keep incomplete records explicitly \
incomplete; don’t infer the raw model reply from the delivered response. A primary category can remain provisional.\nassistant: \
Leave UMEM_LOG_TYPED_REPLY off by default; enable it only for a specific reproduction.";
        let p = best(content, &q("how should incomplete failure records be handled"), |_| 1.0, OPTS).expect("a passage");
        // It also takes the sentence before, which covers one more query word ("failures").
        assert!(p.text.contains("Keep incomplete records explicitly incomplete") && !p.text.contains("selection-first") && !p.text.contains("UMEM"), "{p:?}");
        assert_eq!(p.speaker.as_deref(), Some("user"));
        assert_eq!(content.chars().skip(p.start).take(p.end - p.start).collect::<String>(), p.text);
        assert!(p.matched.contains(&"incomplete".to_string()) && p.matched.contains(&"record".to_string()), "{:?}", p.matched);
    }

    #[test]
    fn rarer_words_decide_and_sentences_join_up_to_the_limit() {
        let content = "user: The weather was fine all week and the weather is fine today and tomorrow too. My bike needs new brakes. \
The brakes squeal on every hill. Anyway the weather is great for a ride in the hills this weekend with friends and family.";
        let idf = |t: &str| if t == "brake" || t == "brakes" { 5.0 } else { 0.5 };
        let p = best(content, &q("weather brakes hill"), idf, Options { max_chars: 400, min_words: 5 }).unwrap();
        assert!(p.text.contains("brakes"), "{p:?}");
        let short = best(content, &q("brakes"), idf, Options { max_chars: 400, min_words: 5 }).unwrap();
        assert_eq!(short.text, "My bike needs new brakes.", "ties go to the shorter passage");
    }

    #[test]
    fn inflected_forms_count_as_the_same_word() {
        let content = format!("user: {} We measured it and you settled on 128 KB. {}", "Unrelated words here. ".repeat(10), "More unrelated words. ".repeat(10));
        let p = best(&content, &q("what size did we settle on"), |_| 1.0, OPTS).unwrap();
        assert_eq!(p.text, "We measured it and you settled on 128 KB.");
    }

    #[test]
    fn only_the_light_stemmers_splits_are_joined() {
        assert_eq!(key("settle"), key("settl"));
        assert_eq!(key("approve"), key("approv"));
        assert_eq!(key("capp"), key("cap"));
        assert_ne!(key("option"), key("optional"));
        // Known side effect: a few unrelated pairs also share a key (plane/plan, hope/hop).
    }

    #[test]
    fn short_records_and_records_without_a_query_word_get_no_passage() {
        assert!(best("user: I like tea.", &q("tea"), |_| 1.0, OPTS).is_none());
        let long = format!("user: {}", "Nothing here is about the topic at all. ".repeat(10));
        assert!(best(&long, &q("penguins"), |_| 1.0, OPTS).is_none());
    }

    #[test]
    fn a_passage_never_spans_two_messages_and_handles_multibyte_text() {
        let content = format!("user: {} Ich mag Äpfel und Birnen sehr.\nassistant: Äpfel sind gesund. {}", "Füller. ".repeat(20), "Mehr Füller. ".repeat(20));
        let p = best(&content, &q("Äpfel Birnen"), |_| 1.0, OPTS).unwrap();
        assert_eq!(p.text, "Ich mag Äpfel und Birnen sehr.");
        assert_eq!(content.chars().skip(p.start).take(p.end - p.start).collect::<String>(), p.text);
    }

    #[test]
    fn cjk_sentences_end_at_their_own_marks() {
        let content = format!("user: {}我们决定每周五在图书馆开会。{}", "今天天气很好。".repeat(30), "明天再说吧。".repeat(30));
        let p = best(&content, &q("图书馆开会"), |_| 1.0, OPTS).unwrap();
        assert_eq!(p.text, "我们决定每周五在图书馆开会。");
    }

    #[test]
    fn an_overlong_sentence_is_cut_at_the_limit() {
        let content = format!("user: {} zebra {}.", "word ".repeat(150), "word ".repeat(150));
        let p = best(&content, &q("zebra"), |_| 1.0, Options { max_chars: 100, min_words: 20 }).unwrap();
        assert!(p.text.chars().count() <= 100 && p.text.contains("zebra"), "{p:?}");
        assert_eq!(content.chars().skip(p.start).take(p.end - p.start).collect::<String>(), p.text);
    }
}
