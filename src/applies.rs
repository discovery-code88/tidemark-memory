//! When a statement applies (opt-in, `TIDEMARK_APPLIES=on`, per request `applies`): a record's date says when
//! something was said, not when it holds. "From next month I start working night shifts", said on 10 August, holds
//! from September. For the user's sentences with an explicit marker, Search adds the resolved period next to the
//! record: a line `[applies: "from next month" = from September 2023]` and a structured `applies` field.
//!
//! Nothing is stored and no record is changed: the period is computed from the sentence and its record's date each
//! time, so it restates what they already say. It does not decide what is true now, and it does not say whether a
//! plan was carried out.
//!
//! Explicit forms only:
//! - a start: from / starting / beginning / as of / effective / since + a time ("since the start of August");
//! - an end: until / till / through + a time; "my last day … will be June 30";
//! - a period: "(for) all of October", "the whole of …", "only this month", "next year only".
//!
//! Precision: a bound keeps the precision of its words. "since January 2025" is the month 2025-01, not a day;
//! "since last year" is the year. Only a named day, or "the start/end of" a month, gives a day.
//!
//! Boundaries:
//! - `from` is the first day, month or year the statement applies; `until` is the last one it certainly applies.
//! - "until February" and "until June 30" do not say whether February, or the 30th, is included. `until` is then
//!   the unit before the named time and `until_latest` the named time; the note says both. Nothing picks one.
//! - "through March", "until the end of April", "my last day … is June 30" and whole periods ("all of October")
//!   include the named time: `until_latest` equals `until`.
//! - A sentence may state only one bound; the other stays empty and is not inferred from the record's date.
//!
//! Years: a stated year is kept, also after a comma ("since February 10, 2024") and from the end of a range ("from
//! July 3 to July 10, 2025"). Without one, a month is read as its occurrence nearest to the record's date ("since"
//! reads backwards) and `year_assumed` is set.
//!
//! Abstentions, where the output cannot express what the words leave open: weeks (no week-start convention), "the
//! middle of" a month, "for the next three months" (approximate, endpoints open), a bare "for three months" (it
//! can look backwards or forwards), an offset ("six weeks from today" is a point six weeks on, not a state that
//! starts today) and times that only date an event ("I quit last month"; those keep their time notes).

use crate::{preferences, search};
use chrono::{Datelike, Duration, Months, NaiveDate};
use regex::Regex;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    Day,
    Month,
    Year,
}

/// A day, a month or a year; `date` is its first day.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Point {
    pub date: NaiveDate,
    pub unit: Unit,
}

impl Point {
    pub fn iso(&self) -> String {
        self.date.format(match self.unit {
            Unit::Day => "%Y-%m-%d",
            Unit::Month => "%Y-%m",
            Unit::Year => "%Y",
        }).to_string()
    }

    fn words(&self) -> String {
        self.date.format(match self.unit {
            Unit::Day => "%-d %B %Y",
            Unit::Month => "%B %Y",
            Unit::Year => "%Y",
        }).to_string()
    }

    fn precision(&self) -> &'static str {
        match self.unit {
            Unit::Day => "day",
            Unit::Month => "month",
            Unit::Year => "year",
        }
    }

    /// The unit before this one.
    fn previous(&self) -> Option<Point> {
        let date = match self.unit {
            Unit::Day => self.date - Duration::days(1),
            Unit::Month => self.date.checked_sub_months(Months::new(1))?,
            Unit::Year => self.date.checked_sub_months(Months::new(12))?,
        };
        Some(Point { date, unit: self.unit })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Applies {
    /// The words of the sentence the period was read from, lower-cased.
    pub phrase: String,
    pub from: Option<Point>,
    /// The last day, month or year it certainly applies.
    pub until: Option<Point>,
    /// The last one it may apply, when the words leave open whether the named time is included ("until February");
    /// otherwise the same as `until`.
    pub until_latest: Option<Point>,
    pub year_assumed: bool,
}

impl Applies {
    pub fn text(&self) -> String {
        let open = match (self.until, self.until_latest) {
            (Some(b), Some(l)) if l != b => format!(" (or through {}, if the named time is included)", l.words()),
            _ => String::new(),
        };
        match (self.from, self.until) {
            (Some(a), Some(b)) if a == b => a.words(),
            (Some(a), Some(b)) => format!("{} through {}{open}", a.words(), b.words()),
            (Some(a), None) => format!("from {}", a.words()),
            (None, Some(b)) => format!("through {}{open}", b.words()),
            (None, None) => String::new(),
        }
    }

    pub fn note(&self) -> String {
        format!("[applies: \"{}\" = {}]", self.phrase, self.text())
    }

    pub fn json(&self, sentence: &str) -> serde_json::Value {
        serde_json::json!({
            "phrase": self.phrase,
            "from": self.from.map(|p| p.iso()), "from_precision": self.from.map(|p| p.precision()),
            "until": self.until.map(|p| p.iso()), "until_latest": self.until_latest.map(|p| p.iso()),
            "until_precision": self.until.map(|p| p.precision()),
            "year_assumed": self.year_assumed, "sentence": sentence,
        })
    }
}

/// A parsed time: the day, month or year it names, whether "the start/end of" fixed a day, and how many words it took.
#[derive(Debug, Clone, Copy)]
struct Time {
    point: Point,
    part: bool,
    /// For "the start of": the day before, which ends what runs "until the start of March".
    year_assumed: bool,
    words: usize,
}

fn month_number(word: &str) -> Option<u32> {
    const NAMES: [&str; 12] = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"];
    NAMES.iter().position(|n| *n == word || (word.len() >= 3 && word != "may" && word.len() <= 4 && n.starts_with(word))).map(|i| i as u32 + 1)
}

fn month_end(year: i32, month: u32) -> Option<NaiveDate> {
    Some(NaiveDate::from_ymd_opt(year, month, 1)?.checked_add_months(Months::new(1))? - Duration::days(1))
}

fn day_number(word: &str) -> Option<u32> {
    let digits = word.trim_end_matches(|c: char| c.is_ascii_alphabetic());
    let suffix = &word[digits.len()..];
    if !matches!(suffix, "" | "st" | "nd" | "rd" | "th") {
        return None;
    }
    digits.parse().ok().filter(|d| (1..=31).contains(d))
}

fn year_number(word: &str) -> Option<i32> {
    (word.len() == 4).then(|| word.parse().ok()).flatten().filter(|y| (1900..=2200).contains(y))
}

fn clean(word: &str) -> &str {
    word.trim_matches(|c: char| !c.is_alphanumeric())
}

/// The time at the start of `tail` (lower-case words), read against the record's date. `backwards` reads a month
/// without a year as its latest occurrence that has begun ("since August"); otherwise as the nearest occurrence.
/// `year` supplies the year when the words give none (the end of a range gave it). None for what cannot be
/// expressed: weeks and "the middle of".
fn time(tail: &str, said: NaiveDate, backwards: bool, year: Option<i32>) -> Option<Time> {
    // A word ends the time when punctuation follows it, except a comma before a year ("February 10, 2024").
    let raw = tail.split_whitespace().take(7).collect::<Vec<_>>();
    let mut words = Vec::new();
    for (k, w) in raw.iter().enumerate() {
        words.push(clean(w));
        let year_follows = w.ends_with(',') && raw.get(k + 1).is_some_and(|n| year_number(clean(n)).is_some());
        if w.ends_with([',', '.', ';', '!', '?', ')', ':']) && !year_follows {
            break;
        }
    }
    let at = |i: usize| words.get(i).copied().unwrap_or("");
    let mut i = 0;
    let simple = |date: NaiveDate, unit: Unit, words: usize| Some(Time { point: Point { date, unit }, part: false, year_assumed: false, words });
    match (at(0), at(1)) {
        ("tomorrow", _) => return simple(said + Duration::days(1), Unit::Day, 1),
        ("today", _) => return simple(said, Unit::Day, 1),
        ("yesterday", _) => return simple(said - Duration::days(1), Unit::Day, 1),
        ("now", "on") => return simple(said, Unit::Day, 2),
        ("next" | "this" | "last", "week") => return None,
        (which @ ("next" | "this" | "last"), unit @ ("month" | "year")) => {
            let step: i32 = match which {
                "next" => 1,
                "this" => 0,
                _ => -1,
            };
            return if unit == "month" {
                let absolute = said.year() * 12 + said.month0() as i32 + step;
                simple(NaiveDate::from_ymd_opt(absolute.div_euclid(12), absolute.rem_euclid(12) as u32 + 1, 1)?, Unit::Month, 2)
            } else {
                simple(NaiveDate::from_ymd_opt(said.year() + step, 1, 1)?, Unit::Year, 2)
            };
        }
        _ => {}
    }
    if at(i) == "the" {
        i += 1;
    }
    let mut part = None;
    if let (p @ ("start" | "beginning" | "middle" | "end"), "of") = (at(i), at(i + 1)) {
        if p == "middle" {
            return None;
        }
        part = Some(p == "end");
        i += 2;
    }
    // "May 1st", "1 May", "1st of May", "May", each with an optional year.
    let (month, day) = if let Some(m) = month_number(at(i)) {
        let d = day_number(at(i + 1)).filter(|_| year_number(at(i + 1)).is_none() && part.is_none());
        i += 1 + d.is_some() as usize;
        (m, d)
    } else if let Some(d) = day_number(at(i)).filter(|_| part.is_none()) {
        let skip = (at(i + 1) == "of") as usize;
        let m = month_number(at(i + 1 + skip))?;
        i += 2 + skip;
        (m, Some(d))
    } else {
        return None;
    };
    let stated_year = year_number(at(i));
    i += stated_year.is_some() as usize;
    // "may" is also a verb and a name: as a month it needs a day, a year or a part.
    if month == 5 && day.is_none() && stated_year.is_none() && part.is_none() {
        return None;
    }
    let first = |year: i32| NaiveDate::from_ymd_opt(year, month, day.unwrap_or(1));
    let last = |year: i32| if day.is_some() { first(year) } else { month_end(year, month) };
    let start = match stated_year.or(year) {
        Some(y) => first(y)?,
        None => {
            let options = (-1..=1).filter_map(|k| Some((first(said.year() + k)?, last(said.year() + k)?))).collect::<Vec<_>>();
            if backwards {
                options.into_iter().filter(|(s, _)| *s <= said).max_by_key(|(s, _)| *s)?.0
            } else {
                // Nearest occurrence; a tie goes to the future.
                let distance = |(s, e): &(NaiveDate, NaiveDate)| if said < *s { (*s - said).num_days() * 2 - 1 } else if said > *e { (said - *e).num_days() * 2 } else { -1 };
                options.into_iter().min_by_key(distance)?.0
            }
        }
    };
    let point = match (day, part) {
        (Some(_), _) => Point { date: start, unit: Unit::Day },
        (None, Some(false)) => Point { date: start, unit: Unit::Day },
        (None, Some(true)) => Point { date: month_end(start.year(), month)?, unit: Unit::Day },
        (None, None) => Point { date: start, unit: Unit::Month },
    };
    Some(Time { point, part: part.is_some(), year_assumed: stated_year.is_none() && year.is_none(), words: i })
}

fn phrase(lower: &str, from: usize, tail_start: usize, t: &Time) -> String {
    let tail = lower[tail_start..].split_whitespace().take(t.words).collect::<Vec<_>>().join(" ");
    let head = lower[from..tail_start].split_whitespace().collect::<Vec<_>>().join(" ");
    format!("{head} {tail}").trim_matches(|c: char| !c.is_alphanumeric()).to_string()
}

/// The year stated at the end of a range that starts with `t` ("july 3 to july 10, 2025" → 2025, or 2024 when the
/// start would otherwise fall after the end: "december 28 to january 3, 2025").
fn range_year(tail: &str, t: &Time, said: NaiveDate) -> Option<i32> {
    let mut rest = tail.split_whitespace().skip(t.words);
    if !matches!(clean(rest.next()?), "to" | "until" | "till" | "through") {
        return None;
    }
    let end = time(&rest.collect::<Vec<_>>().join(" "), said, false, None).filter(|e| !e.year_assumed)?;
    let head = tail.split_whitespace().take(t.words).map(clean).collect::<Vec<_>>().join(" ");
    let year = end.point.date.year();
    let start = time(&head, said, false, Some(year))?;
    Some(if start.point.date > end.point.date { year - 1 } else { year })
}

/// The periods a sentence states for itself, read against its record's date.
pub fn extract(sentence: &str, said: NaiveDate) -> Vec<Applies> {
    static START: OnceLock<Regex> = OnceLock::new();
    static OFFSET: OnceLock<Regex> = OnceLock::new();
    static END: OnceLock<Regex> = OnceLock::new();
    static LAST: OnceLock<Regex> = OnceLock::new();
    static ALL: OnceLock<Regex> = OnceLock::new();
    static ONLY: OnceLock<Regex> = OnceLock::new();
    let start = START.get_or_init(|| Regex::new(r"\b(?P<m>from|starting|beginning|as\s+of|effective|since)\s+(?:(?:on|in|from)\s+)?").unwrap());
    // "six weeks from today", "two days from now": a point that far on, not the start of a state.
    let offset = OFFSET.get_or_init(|| Regex::new(r"\b(?:\d+|an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|few|several|couple\s+of|half\s+an?)\s+(?:minute|hour|day|night|week|weekend|month|year)s?\s*$").unwrap());
    let end = END.get_or_init(|| Regex::new(r"\b(?P<m>until|till|til|through)\s+").unwrap());
    let last = LAST.get_or_init(|| Regex::new(r"\blast\s+day\b[^.;!?]{0,40}?\b(?:is|was|will\s+be|on)\s+(?:on\s+)?").unwrap());
    let all = ALL.get_or_init(|| Regex::new(r"\b(?:for\s+)?(?:all|the\s+whole|the\s+entire|the\s+month)\s+of\s+").unwrap());
    let only = ONLY.get_or_init(|| Regex::new(r"\b(?:(?:only|just)\s+(?:for\s+)?(?P<a>(?:this|next)\s+(?:month|year))|(?P<b>(?:this|next)\s+(?:month|year))\s+only)\b").unwrap());
    let lower = sentence.to_lowercase();
    let mut out: Vec<Applies> = Vec::new();
    let mut push = |a: Applies| {
        if (a.from.is_some() || a.until.is_some()) && !out.iter().any(|o| o.from == a.from && o.until == a.until) {
            out.push(a);
        }
    };
    for m in all.find_iter(&lower) {
        if let Some(t) = time(&lower[m.end()..], said, false, None).filter(|t| t.point.unit != Unit::Day) {
            push(Applies { phrase: phrase(&lower, m.start(), m.end(), &t), from: Some(t.point), until: Some(t.point), until_latest: Some(t.point), year_assumed: t.year_assumed });
        }
    }
    for caps in only.captures_iter(&lower) {
        let (Some(whole), Some(span)) = (caps.get(0), caps.name("a").or(caps.name("b"))) else { continue };
        if let Some(t) = time(span.as_str(), said, false, None) {
            push(Applies { phrase: whole.as_str().split_whitespace().collect::<Vec<_>>().join(" "), from: Some(t.point), until: Some(t.point), until_latest: Some(t.point), year_assumed: false });
        }
    }
    for caps in start.captures_iter(&lower) {
        let (m, marker) = (caps.get(0).unwrap(), &caps["m"]);
        if offset.is_match(&lower[..m.start()]) {
            continue;
        }
        let tail = &lower[m.end()..];
        let Some(mut t) = time(tail, said, marker == "since", None) else { continue };
        if t.year_assumed {
            if let Some(better) = range_year(tail, &t, said).and_then(|y| time(tail, said, false, Some(y))) {
                t = better;
            }
        }
        push(Applies { phrase: phrase(&lower, m.start(), m.end(), &t), from: Some(t.point), until: None, until_latest: None, year_assumed: t.year_assumed });
    }
    for caps in end.captures_iter(&lower) {
        let (m, marker) = (caps.get(0).unwrap(), &caps["m"]);
        let Some(t) = time(&lower[m.end()..], said, false, None) else { continue };
        // "through March" and "until the end of March" include March; "until the start of March" ends the day
        // before. "until February" and "until June 30" leave open whether the named time is included: certain up to
        // the unit before it, possible up to the named time.
        let named_first_day = t.part && t.point.date.day() == 1;
        let (until, latest) = if marker == "through" || (t.part && !named_first_day) {
            (Some(t.point), Some(t.point))
        } else if named_first_day {
            (t.point.previous(), t.point.previous())
        } else {
            (t.point.previous(), Some(t.point))
        };
        push(Applies { phrase: phrase(&lower, m.start(), m.end(), &t), from: None, until, until_latest: latest, year_assumed: t.year_assumed });
    }
    for m in last.find_iter(&lower) {
        if let Some(t) = time(&lower[m.end()..], said, false, None).filter(|t| t.point.unit == Unit::Day && !t.part) {
            push(Applies { phrase: phrase(&lower, m.start(), m.end(), &t), from: None, until: Some(t.point), until_latest: Some(t.point), year_assumed: t.year_assumed });
        }
    }
    out
}

/// The user's sentences of a record that state a period, with the period (time notes are not part of a sentence).
pub fn of_record(content: &str, said: NaiveDate) -> Vec<(String, Applies)> {
    static NOTE: OnceLock<Regex> = OnceLock::new();
    let note = NOTE.get_or_init(|| Regex::new(r"\s*\[time: [^\]]*\]").unwrap());
    let mut out = Vec::new();
    for line in search::user_lines(content) {
        let line = note.replace_all(line, "");
        for sentence in preferences::sentences(&line) {
            out.extend(extract(sentence, said).into_iter().map(|a| (sentence.to_string(), a)));
        }
    }
    out
}

/// The UTC date of a record from its `created_at` (RFC 3339).
#[cfg(test)]
pub fn said(created_at: Option<&str>) -> Option<NaiveDate> {
    chrono::DateTime::parse_from_rfc3339(created_at?).ok().map(|d| d.naive_utc().date())
}

/// The sender's calendar day of a record: its `created_at` instant shifted by the offset its date header states
/// ("… UTC+02:00]"), or the UTC day when the header states none.
pub fn said_in(content: &str, created_at: Option<&str>) -> Option<NaiveDate> {
    static OFFSET: OnceLock<Regex> = OnceLock::new();
    let offset = OFFSET.get_or_init(|| Regex::new(r"^\[[^\]\n]* UTC([+-])(\d\d):(\d\d)\]").unwrap());
    let instant = chrono::DateTime::parse_from_rfc3339(created_at?).ok()?.naive_utc();
    let minutes = offset.captures(content).map_or(0, |c| {
        let m = c[2].parse::<i64>().unwrap_or(0) * 60 + c[3].parse::<i64>().unwrap_or(0);
        if &c[1] == "-" { -m } else { m }
    });
    Some((instant + Duration::minutes(minutes)).date())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(s: &str) -> NaiveDate {
        NaiveDate::parse_from_str(s, "%Y-%m-%d").unwrap()
    }

    /// (from, until, until_latest) of the single period a sentence states.
    fn one(sentence: &str, said: &str) -> (String, String, String) {
        let found = extract(sentence, d(said));
        assert_eq!(found.len(), 1, "{sentence}: {found:?}");
        let iso = |p: Option<Point>| p.map(|p| p.iso()).unwrap_or_default();
        (iso(found[0].from), iso(found[0].until), iso(found[0].until_latest))
    }

    fn t(a: &str, b: &str, c: &str) -> (String, String, String) {
        (a.into(), b.into(), c.into())
    }

    fn none(sentence: &str, said: &str) {
        assert!(extract(sentence, d(said)).is_empty(), "{sentence}: {:?}", extract(sentence, d(said)));
    }

    #[test]
    fn starts_keep_the_precision_of_their_words() {
        assert_eq!(one("From next month I start working night shifts at the warehouse.", "2023-08-10"), t("2023-09", "", ""));
        assert_eq!(one("Starting May 1st I'm going fully vegan, no animal products.", "2023-04-20"), t("2023-05-01", "", ""));
        assert_eq!(one("I've gone vegetarian since the start of August, no more meat for me.", "2023-08-03"), t("2023-08-01", "", ""));
        assert_eq!(one("From now on, call me Sam.", "2023-02-02"), t("2023-02-02", "", ""));
        assert_eq!(one("As of 1 March 2024 I work part time.", "2023-11-30"), t("2024-03-01", "", ""));
        assert_eq!(one("I have been off sugar since December.", "2023-02-10"), t("2022-12", "", ""), "since reads backwards; a month stays a month");
        assert_eq!(one("Starting in January I'll take the early train.", "2023-11-20"), t("2024-01", "", ""), "the nearest January");
        assert_eq!(one("My brother has worked nights since January 2025.", "2026-06-15"), t("2025-01", "", ""), "no start day is invented");
        assert_eq!(one("I've been away since last year, but I forget the month.", "2026-06-15"), t("2025", "", ""), "no 1 January is invented");
        assert_eq!(extract("My brother has worked nights since January 2025.", d("2026-06-15"))[0].note(), "[applies: \"since january 2025\" = from January 2025]");
    }

    #[test]
    fn a_stated_year_is_kept_across_a_comma_and_a_range() {
        let a = &extract("I've owned this bike since February 10, 2024.", d("2026-06-15"))[0];
        assert_eq!((a.from.unwrap().iso(), a.year_assumed, a.phrase.as_str()), ("2024-02-10".to_string(), false, "since february 10, 2024"));
        assert_eq!(one("Starting July 1, 2026, I'll work from home.", "2026-06-15"), t("2026-07-01", "", ""));
        assert!(!extract("Starting July 1, 2026, I'll work from home.", d("2026-06-15"))[0].year_assumed);
        assert_eq!(one("I stayed with Ana from July 3 to July 10, 2025.", "2026-06-15"), t("2025-07-03", "", ""), "the range's year, not the nearest July");
        assert_eq!(one("We were away from December 28 to January 3, 2025.", "2026-06-15"), t("2024-12-28", "", ""), "a start after the end belongs to the year before");
        assert_eq!(one("I'll stay with Ana from July 3 to July 10.", "2026-06-15"), t("2026-07-03", "", ""));
        assert!(extract("I'll stay with Ana from July 3 to July 10.", d("2026-06-15"))[0].year_assumed);
        assert_eq!(one("Starting May 1st, I'm vegan.", "2023-04-20"), t("2023-05-01", "", ""), "a comma that is not before a year still ends the time");
    }

    #[test]
    fn an_offset_is_not_a_start() {
        none("I started the course yesterday; exactly six weeks from today, I plan to finish.", "2026-06-15");
        none("Two days from now on I will know.", "2026-06-15");
        none("The exam is a month from tomorrow.", "2026-06-15");
        assert_eq!(one("From today I'm off caffeine.", "2026-06-15"), t("2026-06-15", "", ""));
    }

    #[test]
    fn ends_say_whether_the_named_time_is_included() {
        assert_eq!(one("Doing dry January, not a drop of alcohol until February.", "2023-01-02"), t("", "2023-01", "2023-02"));
        assert_eq!(
            extract("Doing dry January, not a drop of alcohol until February.", d("2023-01-02"))[0].note(),
            "[applies: \"until february\" = through January 2023 (or through February 2023, if the named time is included)]"
        );
        assert_eq!(one("The offer is open until June 30.", "2023-06-01"), t("", "2023-06-29", "2023-06-30"));
        assert_eq!(one("Until July 1, 2026, I'll work from home.", "2026-06-15"), t("", "2026-06-30", "2026-07-01"));
        assert_eq!(one("My last day at the bank will be June 30, then I'm retired!", "2023-03-15"), t("", "2023-06-30", "2023-06-30"));
        assert_eq!(one("I'm on call through March.", "2023-02-20"), t("", "2023-03", "2023-03"));
        assert_eq!(one("The diet lasts until the end of April.", "2023-03-01"), t("", "2023-04-30", "2023-04-30"));
        assert_eq!(one("I'm covering until the start of March.", "2023-02-01"), t("", "2023-02-28", "2023-02-28"));
    }

    #[test]
    fn whole_periods() {
        assert_eq!(one("No coffee for me this month, I'm doing a caffeine detox for all of October.", "2023-10-01"), t("2023-10", "2023-10", "2023-10"));
        assert_eq!(extract("A caffeine detox for all of October.", d("2023-10-01"))[0].note(), "[applies: \"for all of october\" = October 2023]");
        assert_eq!(one("I'm working from home this month only.", "2023-05-10"), t("2023-05", "2023-05", "2023-05"));
    }

    #[test]
    fn what_the_output_cannot_express_is_left_out() {
        for s in [
            "I'm working from home this week only.",
            "From next week I'm on nights.",
            "I'm abroad for the next three months.",
            "I'm away from the middle of March.",
            "I go running every morning before work, it's how I start my day.",
            "I've frozen my gym membership for three months while I'm working abroad.",
            "I quit my Spanish lessons last month, I just couldn't keep up with work.",
            "I learned a lot from the guide and walked through the old town until dark.",
            "Starting a blog was hard; from what I read, it may take a while.",
            "Since I enjoy podcasts, I listen from time to time.",
            "I got tips from May, my neighbour, and it may rain.",
            "We drove from Lyon through Bordeaux.",
            "I've worked here since I finished school.",
            "I've been on call since Monday.",
        ] {
            none(s, "2023-06-05");
        }
    }

    #[test]
    fn a_records_sentences_are_read_without_their_time_notes() {
        let content = "[Thursday 10 August 2023, 12:00] user: Work is fine. From next month I start working night shifts. [time: next month = September 2023]\nassistant: From next week you should rest.";
        let found = of_record(content, d("2023-08-10"));
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].0, "From next month I start working night shifts.");
        assert_eq!(found[0].1.note(), "[applies: \"from next month\" = from September 2023]");
        assert_eq!(said(Some("2023-08-10T12:00:00Z")), Some(d("2023-08-10")));
        assert_eq!(said(None), None);
        assert_eq!(said_in("[Thursday 1 January 2026, 00:30 UTC+02:00] user: Hi.", Some("2025-12-31T22:30:00Z")), Some(d("2026-01-01")), "the sender's day");
        assert_eq!(said_in("[Thursday 31 December 2026, 23:30 UTC-05:00] \nuser: Hi.", Some("2027-01-01T04:30:00Z")), Some(d("2026-12-31")));
        assert_eq!(said_in("[Wednesday 31 December 2025, 22:30] user: Hi. It was UTC+02:00] there.", Some("2025-12-31T22:30:00Z")), Some(d("2025-12-31")), "no offset in the header: UTC");
    }
}
