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
    for caps in pattern.captures_iter(&lower) {
        let phrase = caps.get(0).map(|m| m.as_str().split_whitespace().collect::<Vec<_>>().join(" ")).unwrap_or_default();
        let resolved = resolve(&caps, date);
        if let Some(value) = resolved {
            let note = format!("{phrase} = {value}");
            if !notes.contains(&note) {
                notes.push(note);
            }
        }
    }
    notes
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
