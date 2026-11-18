//! Evidence assembly (opt-in, `TIDEMARK_ASSEMBLY=on`, per request `assembly`): after the normal ranking, adds records
//! in which the user states a change about a topic of the returned records — a stop ("I dropped the pottery class"),
//! a resumption ("I've re-engaged with organizing…"), a change ("vegetarian since last month", "decided to…"), an
//! exception ("…except on work nights") — so the caller sees the dated statements together. Nothing decides which
//! statement overrides another; each record keeps its own date header.
//!
//! Rules (all from runtime evidence; no labels):
//! - Only the user's own lines can state a change. Assistant replies may be part of a record, never its evidence.
//! - A returned record's topic is its `topic_terms` most characteristic user terms (frequency in the record × rarity,
//!   among terms occurring in at most `rare` of the user's records).
//! - A change relates to a topic when the topic term appears next to the change word, in the same clause (a window
//!   around the cue: "dropped the pottery class", "stopped reading graphic novels").
//! - Additions are groups: the related record and the returned record it relates to (its anchor). The anchor stays.
//! - Protected records are never displaced: anchors, records already added, and returned records that state a change
//!   about one of their own topics.
//! - An addition takes the place of the lowest-ranked unprotected result; if there is none, the group is skipped.
//!   At most `max` groups; none is added just because there is room.

use super::*;

fn cue() -> &'static regex::Regex {
    static CUE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    CUE.get_or_init(|| {
        regex::Regex::new(concat!(
            r"(?i)\b(?:stopped|quit|dropped|gave up|giving up|no longer|not anymore|cancel+ed|stepped back|step back|took a break|",
            r"taking a break|unfollowed|decided against|decided to|i've decided|switched|shifted|instead of|rather than|again|",
            r"re-?engaged|resumed|reinstated|rediscovered|got back into|returned to|from now on|these days|since last|since my|",
            r"ever since|except|unless|only on|not on|started|began)\b",
        ))
        .unwrap()
    })
}

/// The user's sentences in `content` that state a change, stop, resumption or exception (user lines only).
pub fn change_sentences(content: &str) -> Vec<String> {
    search::user_lines(content)
        .into_iter()
        .flat_map(preferences::sentences)
        .filter(|s| cue().is_match(s))
        .map(|s| s.chars().take(600).collect())
        .collect()
}

/// The terms a change sentence is about: those in the same clause as a cue, just before it or after it.
pub fn change_terms(sentence: &str) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    for m in cue().find_iter(sentence) {
        let before = sentence[..m.start()].rsplit([',', ';', '.', '—', '(']).next().unwrap_or("");
        let after = sentence[m.end()..].split([',', ';', '.', '—', ')']).next().unwrap_or("");
        let before = before.split_whitespace().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join(" ");
        let after = after.split_whitespace().take(8).collect::<Vec<_>>().join(" ");
        out.extend(search::tokens(&format!("{before} {after}")));
    }
    out
}

/// An assembled record and the returned record it relates to.
#[derive(Debug, Clone, PartialEq)]
pub struct Group {
    pub record: usize,
    pub anchor: usize,
    pub terms: Vec<String>,
    pub score: f32,
}

/// The ordered results after assembly — (record index, anchor index for assembled records) — and the groups added.
pub fn assemble(records: &[Record], ranked: &[usize], index: &search::Bm25Index, max: usize, rare: f32, topic_terms: usize) -> (Vec<(usize, Option<usize>)>, Vec<Group>) {
    let mut results = ranked.iter().map(|&i| (i, None)).collect::<Vec<_>>();
    if max == 0 || ranked.is_empty() {
        return (results, Vec::new());
    }
    let n = index.len().max(1) as f32;
    let idf = |t: &str| ((n - index.df(t) as f32 + 0.5) / (index.df(t) as f32 + 0.5) + 1.0).ln();
    let is_rare = |t: &str| index.df(t) > 0 && (index.df(t) as f32) / n <= rare;
    let cue_stems = ["stop", "quit", "drop", "gave", "give", "longer", "anymor", "cancel", "step", "back", "break", "decid", "switch",
                     "shift", "instead", "rather", "again", "engag", "resum", "reinstat", "rediscov", "return", "now", "day", "sinc",
                     "except", "unless", "start", "began", "onli", "last", "ever"];
    let topic = |i: usize| -> std::collections::HashSet<String> {
        let mut tf: HashMap<&str, usize> = HashMap::new();
        for t in records[i].user_tokens.iter().filter(|t| is_rare(t) && !cue_stems.iter().any(|m| t.starts_with(m))) {
            *tf.entry(t.as_str()).or_insert(0) += 1;
        }
        let mut weighted = tf.into_iter().map(|(t, f)| (t, f as f32 * idf(t))).collect::<Vec<_>>();
        weighted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(b.0)));
        weighted.into_iter().take(topic_terms).map(|(t, _)| t.to_string()).collect()
    };
    let topics = ranked.iter().map(|&i| (i, topic(i))).collect::<Vec<_>>();
    // Protected: returned records that state a change about one of their own topics.
    let mut protected = topics
        .iter()
        .filter(|(i, t)| records[*i].changes.iter().any(|s| change_terms(s).iter().any(|x| t.contains(x))))
        .map(|(i, _)| *i)
        .collect::<std::collections::HashSet<_>>();
    // Candidate groups: a record not returned whose change sentence is about a returned record's topic.
    let returned = ranked.iter().copied().collect::<std::collections::HashSet<_>>();
    let mut candidates: HashMap<usize, Group> = HashMap::new();
    for (j, r) in records.iter().enumerate() {
        if returned.contains(&j) {
            continue;
        }
        for sentence in &r.changes {
            let about = change_terms(sentence);
            for (anchor, terms) in &topics {
                let mut shared = about.iter().filter(|t| terms.contains(*t)).cloned().collect::<Vec<_>>();
                if shared.is_empty() {
                    continue;
                }
                shared.sort();
                let score = shared.iter().map(|t| idf(t)).sum::<f32>();
                if candidates.get(&j).is_none_or(|g| score > g.score) {
                    candidates.insert(j, Group { record: j, anchor: *anchor, terms: shared, score });
                }
            }
        }
    }
    let mut candidates = candidates.into_values().collect::<Vec<_>>();
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal).then(a.record.cmp(&b.record)));
    let mut added = Vec::new();
    for g in candidates {
        if added.len() >= max {
            break;
        }
        // The anchor must still be in the results.
        if !results.iter().any(|(i, _)| *i == g.anchor) {
            continue;
        }
        // Space: the lowest-ranked result that is neither protected nor this group's anchor; otherwise skip the group.
        let Some(pos) = results.iter().rposition(|(i, _)| !protected.contains(i) && *i != g.anchor) else { continue };
        results.remove(pos);
        let at = results.iter().position(|(i, _)| *i == g.anchor).map_or(results.len(), |p| p + 1);
        results.insert(at, (g.record, Some(g.anchor)));
        protected.insert(g.anchor);
        protected.insert(g.record);
        added.push(g);
    }
    (results, added)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(id: &str, content: &str) -> Record {
        let (user_tokens, instructions, changes) = user_parts(content);
        Record { id: id.into(), speaker: None, content: content.into(), created_at: None, tokens: search::tokens(content), embedding: vec![], user_tokens, instructions, changes }
    }

    fn index(records: &[Record]) -> search::Bm25Index {
        let mut index = search::Bm25Index::default();
        for r in records {
            index.add(&r.user_tokens);
        }
        index
    }

    #[test]
    fn change_sentences_come_from_user_lines_only() {
        let c = "[Mon] \nuser: I love music. I actually dropped the pottery class recently.\nassistant: I stopped by the store again.";
        assert_eq!(change_sentences(c), ["I actually dropped the pottery class recently."]);
        assert!(change_sentences("assistant: You've decided to stop pottery.").is_empty(), "an assistant reply is not the user's statement");
    }

    #[test]
    fn change_terms_are_next_to_the_cue() {
        let t = change_terms("I actually dropped the pottery class recently, which made me sad about my garden.");
        assert!(t.iter().any(|x| x.starts_with("potter")), "{t:?}");
        assert!(!t.iter().any(|x| x.starts_with("garden")), "a term in another clause is not what the change is about");
    }

    #[test]
    fn a_stop_joins_its_anchor_and_protected_records_stay() {
        let records = vec![
            rec("a", "user: I enrolled in a pottery class to meet people. Pottery is fun."),
            rec("b", "user: I stopped reading graphic novels altogether."),
            rec("c", "user: I went to a concert."),
            rec("d", "user: I actually dropped the pottery class recently; the demands were too high."),
            rec("e", "user: I started enjoying graphic novels last year."),
            rec("f", "user: The city is lively."),
            rec("g", "user: Concerts are loud."),
        ];
        let idx = index(&records);
        let (results, added) = assemble(&records, &[0, 1, 2], &idx, 3, 0.5, 8);
        let ids = results.iter().map(|(i, _)| records[*i].id.as_str()).collect::<Vec<_>>();
        assert!(ids.contains(&"b"), "a returned record stating its own change (the graphic-novel stop) is protected: {ids:?}");
        assert!(added.iter().any(|g| records[g.record].id == "d" && records[g.anchor].id == "a"), "the pottery stop joins its anchor: {added:?}");
        for g in &added {
            assert!(results.iter().any(|(i, _)| *i == g.anchor), "every added record keeps its anchor");
        }
        assert_eq!(results.len(), 3, "the number of results does not grow");
        let (results, added) = assemble(&records, &[1], &idx, 3, 0.5, 8);
        assert!(added.is_empty() && results.len() == 1, "a group that does not fit is skipped");
    }
}
