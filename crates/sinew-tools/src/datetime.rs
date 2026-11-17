//! Date and time tools: `current_time`, `date_add` and `date_diff`.
//!
//! Models do not know the current date and are unreliable at calendar
//! arithmetic (month lengths, leap years, daylight saving time). These tools
//! use [jiff](https://docs.rs/jiff) with the IANA time zone database.
//!
//! All three take a [`Clock`] as state: the system clock in production, a
//! fixed instant in tests.
//!
//! ```
//! # async fn demo() -> sinew_core::Result<()> {
//! use serde_json::json;
//! use sinew_core::Tool;
//! use sinew_tools::datetime::{date_add, Clock};
//!
//! let clock = Clock::system();
//! let out = date_add::tool(clock)
//!     .call(json!({"start": "2024-01-31", "months": 1}))
//!     .await?;
//! assert_eq!(out["result"], "2024-02-29");
//! # Ok(()) }
//! ```

/// The date-time library behind these tools, re-exported for [`Clock::now_in`].
pub use jiff;
use jiff::{
    civil::{Date, DateTime},
    tz::TimeZone,
    Span, Timestamp, Unit, ZonedDateTime,
};
use serde::Serialize;

use crate::ToolError;

/// Where "now" comes from, and the time zone used when the model does not
/// name one.
///
/// ```
/// use sinew_tools::datetime::Clock;
///
/// let clock = Clock::fixed(1_794_925_800)?.time_zone("Europe/Lisbon")?;
/// let now = clock.now_in(&sinew_tools::datetime::jiff::tz::TimeZone::UTC);
/// assert_eq!(now.date().to_string(), "2026-11-17");
/// # Ok::<(), sinew_tools::ToolError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Clock {
    fixed: Option<Timestamp>,
    tz: TimeZone,
}

impl Clock {
    /// The system clock and the system time zone (UTC if it cannot be
    /// determined).
    pub fn system() -> Self {
        Self {
            fixed: None,
            tz: TimeZone::system(),
        }
    }

    /// A clock stopped at `unix_seconds`, in UTC (tests, replays).
    pub fn fixed(unix_seconds: i64) -> Result<Self, ToolError> {
        let ts = Timestamp::from_second(unix_seconds)
            .map_err(|e| ToolError::Invalid(format!("bad timestamp: {e}")))?;
        Ok(Self {
            fixed: Some(ts),
            tz: TimeZone::UTC,
        })
    }

    /// Sets the default time zone by IANA name, e.g. `America/Toronto`.
    pub fn time_zone(mut self, name: &str) -> Result<Self, ToolError> {
        self.tz = time_zone(name)?;
        Ok(self)
    }

    /// The current instant in `tz`.
    pub fn now_in(&self, tz: &TimeZone) -> ZonedDateTime {
        self.fixed
            .unwrap_or_else(Timestamp::now)
            .to_zoned_datetime(tz.clone())
    }

    fn zone(&self, name: Option<&str>) -> Result<TimeZone, ToolError> {
        match name.map(str::trim).filter(|n| !n.is_empty()) {
            Some(n) => time_zone(n),
            None => Ok(self.tz.clone()),
        }
    }
}

impl Default for Clock {
    fn default() -> Self {
        Self::system()
    }
}

fn time_zone(name: &str) -> Result<TimeZone, ToolError> {
    if name.eq_ignore_ascii_case("utc") || name == "Z" {
        return Ok(TimeZone::UTC);
    }
    TimeZone::get(name).map_err(|_| {
        ToolError::Invalid(format!(
            "unknown time zone `{name}`; use an IANA name such as America/New_York, Europe/Paris or UTC"
        ))
    })
}

fn zone_name(tz: &TimeZone) -> String {
    tz.iana_name().unwrap_or("UTC").to_string()
}

/// The current date and time, from [`current_time`](fn@current_time).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CurrentTime {
    /// RFC 9557, e.g. `2026-11-17T09:30:00-05:00[America/Toronto]`.
    pub datetime: String,
    pub timezone: String,
    pub date: String,
    pub time: String,
    pub weekday: String,
    /// e.g. `-05:00`.
    pub utc_offset: String,
    pub unix_seconds: i64,
}

/// Get the current date and time. Call it whenever the answer depends on
/// today's date or the current time ("how many days until...", "is it
/// open now"); do not guess the date.
#[sinew_core::tool]
pub fn current_time(
    #[state] clock: Clock,
    /// IANA time zone, e.g. "America/Sao_Paulo", "Europe/Berlin" or "UTC".
    /// Defaults to the application's time zone.
    timezone: Option<String>,
) -> Result<CurrentTime, ToolError> {
    let tz = clock.zone(timezone.as_deref())?;
    let now = clock.now_in(&tz);
    Ok(CurrentTime {
        datetime: now.to_string(),
        timezone: zone_name(&tz),
        date: now.date().to_string(),
        time: now.time().strftime("%H:%M:%S").to_string(),
        weekday: weekday(now.date()),
        utc_offset: now.offset().to_string(),
        unix_seconds: now.timestamp().as_second(),
    })
}

fn weekday(d: Date) -> String {
    d.strftime("%A").to_string()
}

/// A date, a date-time or a zoned date-time parsed from model input.
#[derive(Debug, Clone)]
enum Point {
    Date(Date),
    DateTime(DateTime),
    Zoned(ZonedDateTime),
}

const FORMATS: &str = "YYYY-MM-DD, YYYY-MM-DDTHH:MM[:SS], an RFC 3339 timestamp \
     (2026-11-17T14:00:00Z or with an offset), a zoned value \
     (2026-11-17T14:00[Europe/Paris]), now, today, tomorrow or yesterday";

fn parse_point(s: &str, clock: &Clock, tz: &TimeZone) -> Result<Point, ToolError> {
    let s = s.trim();
    match s.to_ascii_lowercase().as_str() {
        "now" => return Ok(Point::Zoned(clock.now_in(tz))),
        "today" => return Ok(Point::Date(clock.now_in(tz).date())),
        "tomorrow" | "yesterday" => {
            let today = clock.now_in(tz).date();
            let delta = if s.eq_ignore_ascii_case("tomorrow") {
                1
            } else {
                -1
            };
            return today
                .checked_add(Span::new().days(delta))
                .map(Point::Date)
                .map_err(|e| ToolError::Invalid(e.to_string()));
        }
        _ => {}
    }
    if let Ok(z) = s.parse::<ZonedDateTime>() {
        return Ok(Point::Zoned(z));
    }
    if let Ok(ts) = s.parse::<Timestamp>() {
        return Ok(Point::Zoned(ts.to_zoned_datetime(tz.clone())));
    }
    if let Ok(dt) = s.parse::<DateTime>() {
        if s.len() > 10 {
            return Ok(Point::DateTime(dt));
        }
    }
    if let Ok(d) = s.parse::<Date>() {
        return Ok(Point::Date(d));
    }
    Err(ToolError::Invalid(format!(
        "cannot read the date `{s}`; use {FORMATS}"
    )))
}

impl Point {
    fn render(&self) -> (String, String) {
        match self {
            Point::Date(d) => (d.to_string(), weekday(*d)),
            Point::DateTime(dt) => (dt.to_string(), weekday(dt.date())),
            Point::Zoned(z) => (z.to_string(), weekday(z.date())),
        }
    }

    fn to_datetime(&self) -> DateTime {
        match self {
            Point::Date(d) => d.to_datetime(jiff::civil::Time::midnight()),
            Point::DateTime(dt) => *dt,
            Point::Zoned(z) => z.datetime(),
        }
    }

    fn to_zoned(&self, tz: &TimeZone) -> Result<ZonedDateTime, ToolError> {
        match self {
            Point::Zoned(z) => Ok(z.clone()),
            other => other
                .to_datetime()
                .to_zoned_datetime(tz.clone())
                .map_err(|e| ToolError::Invalid(e.to_string())),
        }
    }
}

/// The result of [`date_add`](fn@date_add).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DateResult {
    /// Same form as the input: a date, a date-time, or a zoned date-time
    /// (RFC 9557).
    pub result: String,
    pub weekday: String,
}

fn invalid(e: jiff::Error) -> ToolError {
    ToolError::Invalid(e.to_string())
}

/// Add or subtract a calendar amount to a date or date-time, e.g. "30 days
/// after 2026-01-15" or "3 months before today". Use negative numbers to go
/// back in time. Month and year arithmetic is calendar-aware (2024-01-31 plus
/// 1 month is 2024-02-29), and zoned values account for daylight saving time.
#[sinew_core::tool]
#[allow(clippy::too_many_arguments)]
pub fn date_add(
    #[state] clock: Clock,
    /// Starting point: YYYY-MM-DD, YYYY-MM-DDTHH:MM[:SS], an RFC 3339
    /// timestamp, "now", "today", "tomorrow" or "yesterday".
    start: String,
    years: Option<i64>,
    months: Option<i64>,
    weeks: Option<i64>,
    days: Option<i64>,
    hours: Option<i64>,
    minutes: Option<i64>,
    seconds: Option<i64>,
    /// IANA time zone for "now"/"today" and for timestamps; defaults to the
    /// application's time zone.
    timezone: Option<String>,
) -> Result<DateResult, ToolError> {
    let tz = clock.zone(timezone.as_deref())?;
    let point = parse_point(&start, &clock, &tz)?;
    let span = Span::new()
        .try_years(years.unwrap_or(0))
        .and_then(|s| s.try_months(months.unwrap_or(0)))
        .and_then(|s| s.try_weeks(weeks.unwrap_or(0)))
        .and_then(|s| s.try_days(days.unwrap_or(0)))
        .and_then(|s| s.try_hours(hours.unwrap_or(0)))
        .and_then(|s| s.try_minutes(minutes.unwrap_or(0)))
        .and_then(|s| s.try_seconds(seconds.unwrap_or(0)))
        .map_err(invalid)?;
    let has_time =
        hours.unwrap_or(0) != 0 || minutes.unwrap_or(0) != 0 || seconds.unwrap_or(0) != 0;
    let out = match point {
        Point::Date(d) if !has_time => Point::Date(d.checked_add(span).map_err(invalid)?),
        Point::Date(_) | Point::DateTime(_) => {
            Point::DateTime(point.to_datetime().checked_add(span).map_err(invalid)?)
        }
        Point::Zoned(z) => Point::Zoned(z.checked_add(span).map_err(invalid)?),
    };
    let (result, weekday) = out.render();
    Ok(DateResult { result, weekday })
}

/// The result of [`date_diff`](fn@date_diff).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DateDiff {
    /// ISO 8601 duration from `start` to `end`, e.g. `P1Y2M3DT4H`; negative
    /// (`-P...`) when `end` is before `start`.
    pub duration: String,
    pub years: i64,
    pub months: i64,
    pub days: i64,
    pub hours: i64,
    pub minutes: i64,
    pub seconds: i64,
    /// The whole interval in days (fractional when times are involved).
    pub total_days: f64,
    /// The whole interval in hours.
    pub total_hours: f64,
}

/// Compute the time between two dates or date-times: years, months, days and
/// time, plus the total in days and in hours. Use it for ages, deadlines,
/// "how long until" and "how long ago" questions.
#[sinew_core::tool]
pub fn date_diff(
    #[state] clock: Clock,
    /// Earlier point: YYYY-MM-DD, YYYY-MM-DDTHH:MM[:SS], an RFC 3339
    /// timestamp, "now" or "today".
    start: String,
    /// Later point, same formats.
    end: String,
    /// IANA time zone for "now"/"today" and for mixing dates with
    /// timestamps; defaults to the application's time zone.
    timezone: Option<String>,
) -> Result<DateDiff, ToolError> {
    let tz = clock.zone(timezone.as_deref())?;
    let a = parse_point(&start, &clock, &tz)?;
    let b = parse_point(&end, &clock, &tz)?;
    let (span, total_days, total_hours) = match (&a, &b) {
        (Point::Date(x), Point::Date(y)) => {
            let span = x.until((Unit::Year, *y)).map_err(invalid)?;
            let days = span.total((Unit::Day, *x)).map_err(invalid)?;
            (span, days, days * 24.0)
        }
        (Point::Zoned(_), _) | (_, Point::Zoned(_)) => {
            let x = a.to_zoned(&tz)?;
            let y = b.to_zoned(&tz)?;
            let span = x.until((Unit::Year, &y)).map_err(invalid)?;
            let days = span.total((Unit::Day, &x)).map_err(invalid)?;
            let hours = span.total((Unit::Hour, &x)).map_err(invalid)?;
            (span, days, hours)
        }
        _ => {
            let x = a.to_datetime();
            let y = b.to_datetime();
            let span = x.until((Unit::Year, y)).map_err(invalid)?;
            let days = span.total((Unit::Day, x)).map_err(invalid)?;
            let hours = span.total((Unit::Hour, x)).map_err(invalid)?;
            (span, days, hours)
        }
    };
    Ok(DateDiff {
        duration: span.to_string(),
        years: span.get_years().into(),
        months: span.get_months().into(),
        days: span.get_days().into(),
        hours: span.get_hours().into(),
        minutes: span.get_minutes(),
        seconds: span.get_seconds(),
        total_days,
        total_hours,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};
    use sinew_core::Tool;

    use super::*;

    // 2026-11-17T14:30:00Z, a Tuesday.
    const NOW: i64 = 1_794_925_800;

    fn clock() -> Clock {
        Clock::fixed(NOW).unwrap()
    }

    async fn call(tool: &dyn Tool, args: Value) -> Value {
        tool.call(args).await.unwrap()
    }

    #[tokio::test]
    async fn current_time_in_zones() {
        let t = current_time::tool(clock());
        assert!(!t.side_effects());
        let utc = call(&t, json!({})).await;
        assert_eq!(utc["datetime"], "2026-11-17T14:30:00+00:00[UTC]");
        assert_eq!(utc["weekday"], "Tuesday");
        assert_eq!(utc["unix_seconds"], NOW);
        let sp = call(&t, json!({"timezone": "America/Sao_Paulo"})).await;
        assert_eq!(
            sp["datetime"],
            "2026-11-17T11:30:00-03:00[America/Sao_Paulo]"
        );
        assert_eq!(sp["time"], "11:30:00");
        assert_eq!(sp["utc_offset"], "-03");
        let tokyo = current_time::tool(clock().time_zone("Asia/Tokyo").unwrap());
        let jp = call(&tokyo, json!({})).await;
        assert_eq!(jp["date"], "2026-11-17");
        assert_eq!(jp["time"], "23:30:00");
        assert_eq!(jp["timezone"], "Asia/Tokyo");
        let err = t
            .call(json!({"timezone": "Mars/Olympus"}))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("unknown time zone"), "{err}");
    }

    #[tokio::test]
    async fn adds_calendar_amounts() {
        let t = date_add::tool(clock());
        let r = |v: Value| v["result"].as_str().unwrap().to_string();
        assert_eq!(
            r(call(&t, json!({"start": "2024-01-31", "months": 1})).await),
            "2024-02-29"
        );
        assert_eq!(
            r(call(&t, json!({"start": "2023-01-31", "months": 1})).await),
            "2023-02-28"
        );
        assert_eq!(
            r(call(&t, json!({"start": "2026-11-17", "days": 45})).await),
            "2027-01-01"
        );
        assert_eq!(
            r(call(&t, json!({"start": "2026-03-01", "days": -1})).await),
            "2026-02-28"
        );
        assert_eq!(
            r(call(&t, json!({"start": "today", "weeks": 2})).await),
            "2026-12-01"
        );
        assert_eq!(
            r(call(&t, json!({"start": "tomorrow"})).await),
            "2026-11-18"
        );
        assert_eq!(
            r(call(&t, json!({"start": "2026-11-17", "hours": 36})).await),
            "2026-11-18T12:00:00"
        );
        assert_eq!(
            r(call(&t, json!({"start": "2026-11-17T08:15", "minutes": 50})).await),
            "2026-11-17T09:05:00"
        );
        // DST in New York ends on 2026-11-01: one day later is still 09:00
        // local, but 25 hours have passed.
        assert_eq!(
            r(call(
                &t,
                json!({"start": "2026-10-31T09:00[America/New_York]", "days": 1})
            )
            .await),
            "2026-11-01T09:00:00-05:00[America/New_York]"
        );
        assert_eq!(
            r(call(
                &t,
                json!({"start": "now", "hours": 2, "timezone": "Europe/Paris"})
            )
            .await),
            "2026-11-17T17:30:00+01:00[Europe/Paris]"
        );
        assert_eq!(
            r(call(&t, json!({"start": "2026-11-17T14:00:00Z", "days": 1})).await),
            "2026-11-18T14:00:00+00:00[UTC]"
        );
        let out = call(&t, json!({"start": "2026-11-17", "days": 1})).await;
        assert_eq!(out["weekday"], "Wednesday");
        for bad in [
            json!({"start": "next tuesday"}),
            json!({"start": "2026-13-01"}),
            json!({"start": "2026-11-17", "years": 100000}),
            json!({"start": "2026-11-17", "decades": 1}),
        ] {
            assert!(t.call(bad.clone()).await.is_err(), "{bad}");
        }
    }

    #[tokio::test]
    async fn computes_differences() {
        let t = date_diff::tool(clock());
        let d = call(&t, json!({"start": "2000-02-29", "end": "2026-11-17"})).await;
        assert_eq!(d["duration"], "P26Y8M19D");
        assert_eq!(d["years"], 26);
        assert_eq!(d["months"], 8);
        assert_eq!(d["days"], 19);
        let d = call(&t, json!({"start": "2026-11-17", "end": "2027-01-01"})).await;
        assert_eq!(d["total_days"], 45.0);
        let d = call(&t, json!({"start": "2027-01-01", "end": "2026-11-17"})).await;
        assert_eq!(d["total_days"], -45.0);
        assert!(d["duration"].as_str().unwrap().starts_with('-'));
        let d = call(&t, json!({"start": "now", "end": "2026-12-25"})).await;
        assert_eq!(d["duration"], "P1M7DT9H30M");
        let d = call(
            &t,
            json!({"start": "2026-10-31T12:00[America/New_York]", "end": "2026-11-01T12:00[America/New_York]"}),
        )
        .await;
        assert_eq!(d["total_hours"], 25.0);
        assert_eq!(d["days"], 1);
        let d = call(
            &t,
            json!({"start": "2026-11-17T08:00", "end": "2026-11-17T20:30"}),
        )
        .await;
        assert_eq!(d["total_hours"], 12.5);
        assert!(t
            .call(json!({"start": "soon", "end": "today"}))
            .await
            .is_err());
    }
}
