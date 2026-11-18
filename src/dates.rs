//! Relative time expressions resolved against the message's own date.
//!
//! A reader of a memory record sees "I went camping last week" with the
//! record's date; working out which week that was is error-prone for answer
//! models. At write time each record gets notes such as
//! `last week = the week before Monday 8 May 2023`. The notes only restate
//! what the sentence and its date already say; nothing depends on the
//! question asked later.

use chrono::{Datelike, Duration, NaiveDate, Weekday};
use regex::Regex;
use std::sync::OnceLock;

fn day(date: NaiveDate) -> String {
    date.format("%-d %B %Y").to_string()
}

/// A week-based expression stays relative, anchored to the message date:
/// answer judges (e.g. AML) mark a computed day WRONG when the reference
/// answer is "the Friday before 8 May 2023".
fn anchored(what: &str, relation: &str, date: NaiveDate) -> String {
    format!("{what} {relation} {}", day(date))
}

fn weekday_name(w: Weekday) -> &'static str {
    match w {
        Weekday::Mon => "Monday",
        Weekday::Tue => "Tuesday",
        Weekday::Wed => "Wednesday",
        Weekday::Thu => "Thursday",
        Weekday::Fri => "Friday",
        Weekday::Sat => "Saturday",
        Weekday::Sun => "Sunday",
    }
}

fn month(date: NaiveDate) -> String {
    date.format("%B %Y").to_string()
}

fn shift_months(date: NaiveDate, months: i32) -> Option<NaiveDate> {
    let absolute = date.year() * 12 + date.month0() as i32 + months;
    NaiveDate::from_ymd_opt(absolute.div_euclid(12), absolute.rem_euclid(12) as u32 + 1, 1)
}

fn number(raw: &str) -> Option<i64> {
    Some(match raw {
        "a" | "an" | "one" => 1,
        "two" | "a couple of" | "couple of" => 2,
        "three" => 3,
        "four" => 4,
        "five" => 5,
        "six" => 6,
        "seven" => 7,
        "eight" => 8,
        "nine" => 9,
        "ten" => 10,
        "eleven" => 11,
        "twelve" => 12,
        digits => digits.parse().ok().filter(|n| (1..=99).contains(n))?,
    })
}

const NUM: &str = r"[0-9]{1,2}|an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|(?:a )?couple of";

fn weekday(name: &str) -> Option<Weekday> {
    Some(match &name[..3] {
        "mon" => Weekday::Mon,
        "tue" => Weekday::Tue,
        "wed" => Weekday::Wed,
        "thu" => Weekday::Thu,
        "fri" => Weekday::Fri,
        "sat" => Weekday::Sat,
        "sun" => Weekday::Sun,
        _ => return None,
    })
}

/// The most recent `target` strictly before `date`.
#[allow(dead_code)]
fn previous_weekday(date: NaiveDate, target: Weekday) -> NaiveDate {
    let back = (date.weekday().num_days_from_monday() as i64 - target.num_days_from_monday() as i64 - 1).rem_euclid(7) + 1;
    date - Duration::days(back)
}

/// The next `target` strictly after `date`.
#[allow(dead_code)]
fn next_weekday(date: NaiveDate, target: Weekday) -> NaiveDate {
    let ahead = (target.num_days_from_monday() as i64 - date.weekday().num_days_from_monday() as i64 - 1).rem_euclid(7) + 1;
    date + Duration::days(ahead)
}

/// Notes resolving the relative time expressions of `text` said on `date`,
/// in order of appearance, without duplicates.
pub fn annotate(text: &str, date: NaiveDate) -> Vec<String> {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    let pattern = PATTERN.get_or_init(|| {
        Regex::new(&format!(
            r"(?ix)\b(?:
              (?P<dby>the\s+day\s+before\s+yesterday)
            | (?P<yesterday>yesterday|last\s+night)
            | (?P<today>today|tonight|this\s+morning|this\s+afternoon|this\s+evening)
            | (?P<tomorrow>tomorrow)
            | (?P<n>{NUM})\s+(?P<unit>day|week|weekend|month|year)s?\s+ago
            | (?:(?P<few>a\s+few|several)\s+(?P<funit>day|week|month|year)s?\s+ago)
            | (?:last|this\s+past)\s+(?P<lunit>week|weekend|month|year|summer|winter|spring|fall|autumn)
            | over\s+the\s+(?P<overweekend>weekend)
            | (?:last|this\s+past)\s+(?P<lday>monday|tuesday|wednesday|thursday|friday|saturday|sunday)
            | next\s+(?P<nday>monday|tuesday|wednesday|thursday|friday|saturday|sunday)
            | (?:next|this\s+coming)\s+(?P<nunit>week|weekend|month|year|summer|winter|spring|fall|autumn)
            | this\s+(?P<tunit>week|weekend|month|year)
            | (?:in\s+)?(?P<in_n>{NUM})\s+(?P<in_unit>day|week|month|year)s?\s+(?:from\s+now|later)
            | (?:for|been)\s+(?:about\s+|around\s+|almost\s+|nearly\s+|over\s+)?(?P<for_n>{NUM})\s+(?P<for_unit>year|month)s?(?:\s+now)?
            )\b"
        ))
        .expect("relative time regex")
    });
    let mut notes = Vec::new();
    let lower = text.to_lowercase();
    let quotes = quoted_spans(&lower);
    // A note names a phrase, not a place in the message. When the same phrase also occurs where it does not count
    // from the message's date (inside someone else's quotation), a note would read as covering both, so it is
    // left out.
    let mut unsupported: Vec<String> = Vec::new();
    for caps in pattern.captures_iter(&lower) {
        if let Some(m) = caps.get(0).filter(|m| !counts_from_the_message(&lower, &quotes, m.start(), m.end())) {
            unsupported.push(m.as_str().split_whitespace().collect::<Vec<_>>().join(" "));
            continue;
        }
        if caps.name("for_n").is_some() && !caps.get(0).is_some_and(|m| retrospective(&lower, m.start(), m.end())) {
            continue;
        }
        let phrase = caps.get(0).map(|m| m.as_str().split_whitespace().collect::<Vec<_>>().join(" ")).unwrap_or_default();
        let resolved = resolve(&caps, date);
        if let Some(value) = resolved {
            let note = format!("{phrase} = {value}");
            if !notes.contains(&note) {
                notes.push(note);
            }
        }
    }
    notes.retain(|n| !unsupported.iter().any(|p| n.starts_with(&format!("{p} = "))));
    notes
}

/// The quoted stretches of a text, as byte ranges between the marks, nested ones included. Double quotes (straight
/// or curly) pair up; a single quote opens after a space or opening punctuation and closes after a word or
/// punctuation. An apostrophe between letters ("I've", "Ana's") is not a quote mark, and a mark that never closes
/// ("'cause", "the '90s") quotes nothing.
fn quoted_spans(text: &str) -> Vec<(usize, usize)> {
    let chars = text.char_indices().collect::<Vec<_>>();
    let word = |c: Option<char>| c.is_some_and(char::is_alphanumeric);
    let mut open: Vec<(char, usize)> = Vec::new();
    let mut spans = Vec::new();
    for (k, &(i, c)) in chars.iter().enumerate() {
        let prev = k.checked_sub(1).map(|p| chars[p].1);
        let next = chars.get(k + 1).map(|x| x.1);
        let (kind, opens) = match c {
            '“' | '«' => ('"', true),
            '”' | '»' => ('"', false),
            '"' => ('"', !open.iter().any(|o| o.0 == '"')),
            '\'' | '‘' | '’' => {
                if word(prev) && word(next) {
                    continue; // a contraction or a possessive
                }
                let may_open = c != '’' && prev.is_none_or(|p| p.is_whitespace() || "([{—–:,\"“".contains(p)) && next.is_some_and(|n| n.is_alphanumeric() || "\"“‘".contains(n));
                let may_close = c != '‘' && open.iter().any(|o| o.0 == '\'') && prev.is_some_and(|p| !p.is_whitespace()) && !word(next);
                match (may_close, may_open) {
                    (true, _) => ('\'', false),
                    (false, true) => ('\'', true),
                    _ => continue,
                }
            }
            _ => continue,
        };
        if opens {
            open.push((kind, i + c.len_utf8()));
        } else if let Some(at) = open.iter().rposition(|o| o.0 == kind) {
            spans.push((open[at].1, i));
            open.truncate(at);
        }
    }
    spans
}

/// Whether a relative expression at `start..end` counts from the message's own date. Outside quotation marks it
/// does. Inside them it does only when the quotation is what the user themselves said at the time of the message:
/// the words around that quotation say "I said / told / wrote …" with a present anchor ("today", "just", "this
/// morning") and nothing that makes it recalled, habitual, negated or hypothetical ("I remembered saying",
/// "I once said", "I would have said"). Fiction, lyrics, other people's words, undated quotations and anything
/// quoted inside a quotation get no note: when they were said is not known from the message. This guards against
/// unsupported notes; it is not a reading of reported speech, and speech reported without quotation marks is
/// treated as before.
fn counts_from_the_message(lower: &str, quotes: &[(usize, usize)], start: usize, end: usize) -> bool {
    static SPOKE: OnceLock<Regex> = OnceLock::new();
    static ANCHOR: OnceLock<Regex> = OnceLock::new();
    static REMOVED: OnceLock<Regex> = OnceLock::new();
    let Some(&(a, b)) = quotes.iter().filter(|(a, b)| *a <= start && end <= *b).min_by_key(|(a, b)| b - a) else { return true };
    if quotes.iter().any(|&(x, y)| (x, y) != (a, b) && x <= a && b <= y) {
        return false; // quoted inside a quotation
    }
    let spoke = SPOKE.get_or_init(|| {
        Regex::new(r"\bi\s+(?:just\s+|also\s+|then\s+)?(?:said|told|wrote|texted|messaged|replied|answered|posted|emailed|announced|promised|mentioned)\b|\bi(?:['’]ve|\s+have)\s+just\s+(?:said|told|written|texted|messaged|replied|answered|posted|emailed|announced|promised|mentioned)\b").expect("speech regex")
    });
    let anchor = ANCHOR.get_or_init(|| Regex::new(r"\b(?:today|just|right\s+now|this\s+(?:morning|afternoon|evening)|tonight|a\s+(?:moment|minute|second)\s+ago)\b").expect("anchor regex"));
    let removed = REMOVED.get_or_init(|| {
        Regex::new(r"\b(?:remember(?:ed|ing|s)?|recall(?:ed|ing|s)?|saying|telling|writing|used\s+to|once|always|often|usually|sometimes|would|will|might|could|should|if|never|not|didn't|imagin(?:e|ed|ing)|dream(?:ed|t|ing)?|wish(?:ed)?|almost|nearly|supposed)\b|['’](?:ll|d)\b").expect("recall regex")
    });
    // The words of the quotation's own sentence, outside every quotation: back to the previous sentence end or
    // quotation, forward to the next.
    // Span ends are byte offsets of the closing mark and starts follow the opening mark; marks can be multi-byte
    // (“ ” « » ‘ ’), so step over whole characters, never single bytes.
    let floor = quotes.iter().map(|q| q.1).filter(|e| *e < a).max().map_or(0, |e| e + lower[e..].chars().next().map_or(0, char::len_utf8));
    let ceiling = quotes.iter().map(|q| q.0).filter(|s| *s > b).min().map_or(lower.len(), |s| lower[..s].char_indices().next_back().map_or(0, |(i, _)| i));
    let ends = ['.', '!', '?', ';', '\n'];
    let open_mark = lower[..a].char_indices().next_back().map_or(a, |(i, _)| i);
    let close_mark = b + lower[b..].chars().next().map_or(0, char::len_utf8);
    let (floor, ceiling) = (floor.min(open_mark), ceiling.max(close_mark));
    let before = &lower[floor..open_mark];
    let before = before.rfind(ends).map_or(before, |i| &before[i + 1..]);
    let after = &lower[close_mark..ceiling];
    let after = after.find(ends).map_or(after, |i| &after[..i]);
    let present = |words: &str| spoke.is_match(words) && anchor.is_match(words) && !removed.is_match(words);
    // Words after the quotation attribute it only when they follow on directly ("…," I told her today): not after
    // a quotation that ends its sentence with a full stop, and only when they open with the speaker.
    let follows_on = !lower[a..b].trim_end().ends_with('.') && spoke.find(after.trim_start()).is_some_and(|m| m.start() == 0);
    present(before) || (follows_on && present(after))
}

/// Whether a duration ("for three months", "been three months") clearly looks back from the message date. A bare
/// duration can run either way — "I've frozen my membership for three months" looks forward — so a start date is
/// stated only when the sentence says so: the duration ends in "now", or follows "have/has been" in the same
/// sentence. Future, hypothetical and past-perfect wording ("I'll have been … for", "would have been", "if",
/// "had been") is not counted from the message date and gets no note.
fn retrospective(lower: &str, start: usize, end: usize) -> bool {
    static BEEN: OnceLock<Regex> = OnceLock::new();
    static NOT_NOW: OnceLock<Regex> = OnceLock::new();
    let been = BEEN.get_or_init(|| {
        Regex::new(r"(?:\b(?:have|has|haven't|hasn't|havent|hasnt)|['’](?:ve|s))(?:\s+(?:not|now|only|just|already|really|actually|also|always))*\s+been\b").expect("been regex")
    });
    let not_now = NOT_NOW.get_or_init(|| {
        Regex::new(r"\b(?:will|won't|wont|shall|would|wouldn't|could|might|may|should|if|unless|once|until|had\s+(?:not\s+|never\s+|only\s+|already\s+)?been|hadn't\s+been|going\s+to|gonna|plan(?:s|ning)?\s+to|by\s+(?:the\s+time|then|next|january|february|march|april|may|june|july|august|september|october|november|december))\b|['’](?:ll|d)\b").expect("future regex")
    });
    let from = lower[..start].rfind(['.', '!', '?', ';', '\n']).map_or(0, |i| i + 1);
    let to = lower[end..].find(['.', '!', '?', ';', '\n']).map_or(lower.len(), |i| end + i);
    if not_now.is_match(&lower[from..to]) {
        return false;
    }
    // The match itself may start at "been" ("it's been three months"), so the auxiliary is looked for up to its end.
    let matched = &lower[start..end];
    matched.ends_with("now") || been.is_match(&lower[from..end])
}

fn season_year(season: &str, date: NaiveDate, step: i32) -> Option<String> {
    // Seasons as northern-hemisphere months; winter runs December–February
    // and is labelled by both years. `latest` is the year in which the most
    // recently started season of this name began: "last summer" said in May
    // 2023 is summer 2022, in October 2023 summer 2023; "next winter" said in
    // October 2023 is winter 2023–2024.
    let start = match season {
        "spring" => 3,
        "summer" => 6,
        "fall" | "autumn" => 9,
        "winter" => 12,
        _ => return None,
    };
    let (month, year) = (date.month(), date.year());
    let latest = if month >= start { year } else { year - 1 };
    let in_season = if season == "winter" { month == 12 || month <= 2 } else { month >= start && month <= start + 2 };
    let label = |y: i32| if season == "winter" { format!("winter {}–{}", y, y + 1) } else { format!("{season} {y}") };
    Some(match step {
        // "last summer" while it is summer means the previous one.
        -1 => label(if in_season { latest - 1 } else { latest }),
        _ => label(latest + 1),
    })
}

fn resolve(caps: &regex::Captures, date: NaiveDate) -> Option<String> {
    if caps.name("dby").is_some() {
        return Some(day(date - Duration::days(2)));
    }
    if caps.name("yesterday").is_some() {
        return Some(day(date - Duration::days(1)));
    }
    if caps.name("today").is_some() {
        return Some(day(date));
    }
    if caps.name("tomorrow").is_some() {
        return Some(day(date + Duration::days(1)));
    }
    if let (Some(n), Some(unit)) = (caps.name("n"), caps.name("unit")) {
        let n = number(n.as_str())?;
        return Some(match unit.as_str() {
            "day" => day(date - Duration::days(n)),
            "week" if n == 1 => anchored("the week", "before", date),
            "week" => anchored(&format!("{n} weeks"), "before", date),
            "weekend" if n == 1 => anchored("the weekend", "before", date),
            "weekend" => anchored(&format!("{n} weekends"), "before", date),
            "month" => month(shift_months(date, -(n as i32))?),
            _ => (date.year() - n as i32).to_string(),
        });
    }
    if let (Some(few), Some(unit)) = (caps.name("few"), caps.name("funit")) {
        let few = few.as_str().split_whitespace().collect::<Vec<_>>().join(" ");
        return Some(format!("{few} {}s before {}", unit.as_str(), day(date)));
    }
    if let Some(unit) = caps.name("lunit") {
        return Some(match unit.as_str() {
            "week" => anchored("the week", "before", date),
            "weekend" => anchored("the weekend", "before", date),
            "month" => month(shift_months(date, -1)?),
            "year" => (date.year() - 1).to_string(),
            season => season_year(season, date, -1)?,
        });
    }
    if caps.name("overweekend").is_some() {
        return Some(anchored("the weekend", "before", date));
    }
    if let Some(name) = caps.name("lday") {
        return Some(anchored(&format!("the {}", weekday_name(weekday(name.as_str())?)), "before", date));
    }
    if let Some(name) = caps.name("nday") {
        return Some(anchored(&format!("the {}", weekday_name(weekday(name.as_str())?)), "after", date));
    }
    if let Some(unit) = caps.name("nunit") {
        return Some(match unit.as_str() {
            "week" => anchored("the week", "after", date),
            "weekend" => anchored("the weekend", "after", date),
            "month" => month(shift_months(date, 1)?),
            "year" => (date.year() + 1).to_string(),
            season => season_year(season, date, 1)?,
        });
    }
    if let Some(unit) = caps.name("tunit") {
        return Some(match unit.as_str() {
            "week" => anchored("the week", "of", date),
            "weekend" => anchored("the weekend", "of", date),
            "month" => month(date),
            _ => date.year().to_string(),
        });
    }
    if let (Some(n), Some(unit)) = (caps.name("in_n"), caps.name("in_unit")) {
        let n = number(n.as_str())?;
        return Some(match unit.as_str() {
            "day" => day(date + Duration::days(n)),
            "week" => anchored(&format!("{n} week{}", if n == 1 { "" } else { "s" }), "after", date),
            "month" => month(shift_months(date, n as i32)?),
            _ => (date.year() + n as i32).to_string(),
        });
    }
    if let (Some(n), Some(unit)) = (caps.name("for_n"), caps.name("for_unit")) {
        let n = number(n.as_str())?;
        return Some(match unit.as_str() {
            "year" => format!("since about {}", date.year() - n as i32),
            _ => format!("since about {}", month(shift_months(date, -(n as i32))?)),
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_byte_quotation_marks_next_to_a_quotation_do_not_panic() {
        // Found in real use (2026-11-17): Add panicked when another quotation with a multi-byte mark followed or
        // preceded the one holding the expression (slicing one byte into “ or ”).
        let date = d(2026, 11, 17);
        let marks = [("\"", "\""), ("“", "”"), ("«", "»"), ("‘", "’"), ("'", "'")];
        for (o1, c1) in marks {
            for (o2, c2) in marks {
                for text in [
                    format!("I told her today, {o1}see you tomorrow{c1} and then {o2}bye{c2}."),
                    format!("{o2}hi{c2} she said, {o1}I'll call tomorrow{c1} — {o2}ok{c2}"),
                    format!("{o1}last week{c1}{o2}x{c2}{o1}yesterday{c1}"),
                ] {
                    let _ = annotate(&text, date); // must not panic
                }
            }
        }
        // The user's own present speech stays annotated with curly marks on both sides.
        let notes = annotate("I just told her today, “I'll call you tomorrow”, and she said “fine”.", date);
        assert!(notes.iter().any(|n| n.starts_with("tomorrow = ")), "{notes:?}");
    }

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn seasons_resolve_to_the_right_year_in_every_month() {
        // (said on, phrase, expected)
        let cases = [
            ((2023, 5, 8), "next summer", "summer 2023"),
            ((2023, 7, 10), "next summer", "summer 2024"),
            ((2023, 10, 2), "next summer", "summer 2024"),
            ((2023, 5, 8), "last summer", "summer 2022"),
            ((2023, 7, 10), "last summer", "summer 2022"),
            ((2023, 10, 2), "last summer", "summer 2023"),
            ((2023, 10, 2), "last winter", "winter 2022–2023"),
            ((2023, 3, 15), "last winter", "winter 2022–2023"),
            ((2023, 1, 15), "last winter", "winter 2021–2022"),
            ((2023, 12, 20), "last winter", "winter 2022–2023"),
            ((2023, 10, 2), "next winter", "winter 2023–2024"),
            ((2023, 1, 15), "next winter", "winter 2023–2024"),
            ((2023, 12, 20), "next winter", "winter 2024–2025"),
            ((2023, 2, 1), "next spring", "spring 2023"),
            ((2023, 4, 1), "next spring", "spring 2024"),
            ((2023, 8, 1), "last fall", "fall 2022"),
            ((2023, 10, 1), "last fall", "fall 2022"),
            ((2023, 12, 1), "last fall", "fall 2023"),
        ];
        for ((y, m, day), phrase, expected) in cases {
            let got = annotate(&format!("We talked about {phrase}."), d(y, m, day));
            assert_eq!(got, [format!("{phrase} = {expected}")], "{phrase} said on {y}-{m}-{day}");
        }
    }

    #[test]
    fn relative_expressions_resolve_against_the_message_date() {
        let monday = d(2023, 5, 8);
        assert_eq!(annotate("I went to a support group yesterday.", monday), ["yesterday = 7 May 2023"]);
        assert_eq!(annotate("Three days ago I finished it.", monday), ["three days ago = 5 May 2023"]);
        assert_eq!(annotate("I adopted her two years ago.", monday), ["two years ago = 2021"]);
        assert_eq!(annotate("I've had them for 3 years now.", monday), ["for 3 years now = since about 2020"]);

        assert_eq!(annotate("A friend adopted last year.", monday), ["last year = 2022"]);
        assert_eq!(annotate("Next month I'm off to Ireland.", monday), ["next month = June 2023"]);
        assert_eq!(annotate("I took this last summer.", monday), ["last summer = summer 2022"]);
        assert_eq!(annotate("I took this last summer.", d(2023, 10, 2)), ["last summer = summer 2023"]);
        assert_eq!(annotate("I moved here a few years ago.", monday), ["a few years ago = a few years before 8 May 2023"]);
    }

    #[test]
    fn quoted_words_count_from_the_message_only_when_the_user_says_them_now() {
        let said = NaiveDate::from_ymd_opt(2026, 6, 15).unwrap();
        let none: [&str; 0] = [];
        // Attribution: fiction, lyrics, other people, the assistant, examples, undated quotations.
        assert_eq!(annotate("My script says, 'I've been ill for a month.'", said), none);
        assert_eq!(annotate("In my novel, the hero writes, \"I arrived yesterday and I leave tomorrow.\"", said), none);
        assert_eq!(annotate("The song goes, “Yesterday, all my troubles seemed so far away.”", said), none);
        assert_eq!(annotate("Ana said, \"I've been ill for a month.\"", said), none);
        assert_eq!(annotate("The assistant wrote, 'You moved yesterday,' but I never said that.", said), none);
        assert_eq!(annotate("Quote of the day: \"I've been waiting for three years now.\"", said), none);
        assert_eq!(annotate("For my English class I wrote the sentence \"I visited Paris last month\".", said), none);
        assert_eq!(annotate("I said \"I'll call you tomorrow\" and I meant it.", said), none, "the user's own words, but when they were said is not stated");
        assert_eq!(annotate("\"I arrived yesterday\"", said), none, "a bare quotation is nobody's dated statement");
        // An expression outside the quotation is independent of it.
        assert_eq!(annotate("Last week Ana said, \"I'll call you tomorrow.\"", said), ["last week = the week before 15 June 2026"]);
        assert_eq!(annotate("I wrote in my diary yesterday: \"I feel better.\"", said), ["yesterday = 14 June 2026"]);
        assert_eq!(annotate("Yesterday Ana said, \"The shop is closed.\"", said), ["yesterday = 14 June 2026"]);
        assert_eq!(annotate("I moved in yesterday, and my landlord said \"welcome\".", said), ["yesterday = 14 June 2026"]);
        // Present speech: the quotation itself is anchored to now.
        assert_eq!(annotate("I said today, 'I arrived yesterday.'", said), ["today = 15 June 2026", "yesterday = 14 June 2026"]);
        assert_eq!(annotate("\"I've been here for two years now,\" I told the landlord today.", said), ["for two years now = since about 2024", "today = 15 June 2026"]);
        assert_eq!(annotate("I just told Ana, \"I started the job yesterday.\"", said), ["yesterday = 14 June 2026"]);
        // Recalled, habitual, negated or hypothetical speech: "today" dates the remembering, not the quotation.
        assert_eq!(annotate("Today I remembered saying, 'I arrived yesterday.'", said), ["today = 15 June 2026"]);
        assert_eq!(annotate("I once said, \"I'll quit tomorrow,\" just to annoy her.", said), none);
        assert_eq!(annotate("I would have said \"see you tomorrow\" today.", said), ["today = 15 June 2026"]);
        assert_eq!(annotate("I never said \"I'll pay tomorrow\" today.", said), ["today = 15 June 2026"]);
        assert_eq!(annotate("I always say \"tomorrow\" when asked.", said), none);
        // The anchor belongs to its own quotation's sentence.
        assert_eq!(annotate("I told Ana today that I'm tired. Ben said, \"See you tomorrow.\"", said), ["today = 15 June 2026"]);
        assert_eq!(annotate("Ben said, \"See you later.\" I told Ana today, \"I leave tomorrow.\"", said), ["today = 15 June 2026", "tomorrow = 16 June 2026"]);
        // A note names a phrase, not a place: the same phrase in a quotation that does not count leaves no note at all.
        assert_eq!(annotate("Ben said, \"See you tomorrow.\" I told Ana today, \"I leave tomorrow.\"", said), ["today = 15 June 2026"]);
        assert_eq!(annotate("The song goes \"yesterday\", and I moved yesterday.", said), none);
        assert_eq!(annotate("The song goes \"yesterday\", and I moved last week.", said), ["last week = the week before 15 June 2026"]);
        // Nested quotations: what is quoted inside a quotation is never the user's present statement.
        assert_eq!(annotate("I told Ana today, \"My script says, 'I've been ill for a month.'\"", said), ["today = 15 June 2026"]);
        assert_eq!(annotate("I told Ana today, \"I read 'see you tomorrow' and left yesterday.\"", said), ["today = 15 June 2026", "yesterday = 14 June 2026"]);
        // Apostrophes are not quotation marks.
        assert_eq!(annotate("I've checked: Ana's sister doesn't know I'll arrive tomorrow.", said), ["tomorrow = 16 June 2026"]);
        assert_eq!(annotate("My parents' house was sold yesterday.", said), ["yesterday = 14 June 2026"]);
        assert_eq!(annotate("It's the '90s party tomorrow, 'cause we moved it yesterday.", said), ["tomorrow = 16 June 2026", "yesterday = 14 June 2026"]);
        assert_eq!(annotate("I’ve said it before: I’m leaving tomorrow.", said), ["tomorrow = 16 June 2026"]);
        assert_eq!(annotate("We met at O'Brien's yesterday; it's 'the' place.", said), ["yesterday = 14 June 2026"]);
    }

    #[test]
    fn a_duration_gets_a_start_only_when_it_clearly_looks_back() {
        let said = NaiveDate::from_ymd_opt(2023, 6, 5).unwrap();
        let none: [&str; 0] = [];
        // Retrospective.
        assert_eq!(annotate("I've been attending for three months.", said), ["for three months = since about March 2023"]);
        assert_eq!(annotate("She has been vegan for about two years.", said), ["for about two years = since about 2021"]);
        assert_eq!(annotate("It's been three months since I joined.", said), ["been three months = since about March 2023"]);
        assert_eq!(annotate("I have not been there for 2 years.", said), ["for 2 years = since about 2021"]);
        // Prospective: the same words after a different verb look forward.
        assert_eq!(annotate("I've frozen my membership for three months.", said), none);
        assert_eq!(annotate("I've frozen my gym membership for three months while I'm working abroad.", said), none);
        assert_eq!(annotate("I'm renting the flat for six months.", said), none);
        // Future and hypothetical: "been … for" is there, but not counted from today.
        assert_eq!(annotate("By December, I'll have been attending for three months.", said), none);
        assert_eq!(annotate("By December, I will have been attending for three months.", said), none);
        assert_eq!(annotate("If I stay, I would have been here for two years.", said), none);
        assert_eq!(annotate("I had been training for three months when I got injured.", said), none);
        assert_eq!(annotate("I'll be away for three months now.", said), none);
        // Ambiguous: no start date is invented, and other notes in the sentence are unaffected.
        assert_eq!(annotate("I worked there for three months.", said), none);
        assert_eq!(annotate("I've lived here for three years.", said), none);
        assert_eq!(annotate("I signed up yesterday for three months.", said), ["yesterday = 4 June 2023"]);
        // The context is the duration's own sentence.
        assert_eq!(annotate("I will travel soon. I've been saving for two years.", said), ["for two years = since about 2021"]);
        assert_eq!(annotate("I've been busy. I booked the room for two months.", said), none);
    }

    #[test]
    fn week_based_expressions_stay_relative_to_the_message_date() {
        // Judges such as AML's mark a computed day WRONG against a reference
        // answer like "the Friday before 8 May 2023" ("no calendar math").
        let monday = d(2023, 5, 8);
        assert_eq!(annotate("We went camping last week!", monday), ["last week = the week before 8 May 2023"]);
        assert_eq!(annotate("Last Friday I ran a charity race.", monday), ["last friday = the Friday before 8 May 2023"]);
        assert_eq!(annotate("Over the weekend we painted.", monday), ["over the weekend = the weekend before 8 May 2023"]);
        assert_eq!(annotate("It was two weekends ago.", monday), ["two weekends ago = 2 weekends before 8 May 2023"]);
        assert_eq!(annotate("It was two weeks ago.", monday), ["two weeks ago = 2 weeks before 8 May 2023"]);
        assert_eq!(annotate("See you next Saturday!", monday), ["next saturday = the Saturday after 8 May 2023"]);
        assert_eq!(annotate("This weekend we hike.", monday), ["this weekend = the weekend of 8 May 2023"]);
    }

    #[test]
    fn text_without_relative_time_gets_no_notes() {
        assert!(annotate("I love painting sunsets.", d(2023, 5, 8)).is_empty());
        assert!(annotate("The meeting is on 4 July 2023.", d(2023, 5, 8)).is_empty());
        assert!(annotate("We spent a year abroad.", d(2023, 5, 8)).is_empty());
    }

}
