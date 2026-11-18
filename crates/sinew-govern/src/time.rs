//! [`TimeWindow`]: days of the week and a time range, at a fixed UTC
//! offset.

use serde::{Deserialize, Serialize};

use crate::GovernError;

/// A day of the week.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Weekday {
    #[serde(alias = "monday")]
    Mon,
    #[serde(alias = "tuesday")]
    Tue,
    #[serde(alias = "wednesday")]
    Wed,
    #[serde(alias = "thursday")]
    Thu,
    #[serde(alias = "friday")]
    Fri,
    #[serde(alias = "saturday")]
    Sat,
    #[serde(alias = "sunday")]
    Sun,
}

impl Weekday {
    /// Monday to Friday.
    pub const WORKDAYS: [Weekday; 5] = [
        Weekday::Mon,
        Weekday::Tue,
        Weekday::Wed,
        Weekday::Thu,
        Weekday::Fri,
    ];

    fn from_days_since_epoch(days: i64) -> Self {
        // 1970-01-01 was a Thursday.
        match (days + 3).rem_euclid(7) {
            0 => Weekday::Mon,
            1 => Weekday::Tue,
            2 => Weekday::Wed,
            3 => Weekday::Thu,
            4 => Weekday::Fri,
            5 => Weekday::Sat,
            _ => Weekday::Sun,
        }
    }
}

/// Days of the week and a `[from, to)` time range, evaluated at a fixed
/// offset from UTC (no daylight-saving rules: use the offset that applies,
/// or UTC). A range whose end is before its start spans midnight
/// (`22:00`–`06:00`); the day is the one the range starts on.
///
/// Used with [`Rule::during`](crate::Rule::during) and
/// [`Rule::outside`](crate::Rule::outside).
///
/// ```
/// use sinew_govern::TimeWindow;
///
/// let hours = TimeWindow::workdays("09:00", "17:00").unwrap().utc_offset("-05:00").unwrap();
/// // 2026-11-16 (a Monday) 15:00 UTC is 10:00 at UTC-5.
/// assert!(hours.contains(1_794_841_200_000));
/// // 2026-11-15 (a Sunday).
/// assert!(!hours.contains(1_794_754_800_000));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeWindow {
    /// Days the window is open (empty: every day).
    #[serde(default)]
    pub days: Vec<Weekday>,
    /// Start, minutes after midnight.
    pub from_minute: u16,
    /// End (exclusive), minutes after midnight; `from == to` means all day.
    pub to_minute: u16,
    /// Offset from UTC in minutes (e.g. `-300` for UTC-5).
    #[serde(default)]
    pub utc_offset_minutes: i16,
}

impl TimeWindow {
    /// Every day between `from` and `to` (`HH:MM`).
    pub fn daily(from: &str, to: &str) -> Result<Self, GovernError> {
        Ok(Self {
            days: Vec::new(),
            from_minute: parse_hhmm(from)?,
            to_minute: parse_hhmm(to)?,
            utc_offset_minutes: 0,
        })
    }

    /// Monday to Friday between `from` and `to` (`HH:MM`).
    pub fn workdays(from: &str, to: &str) -> Result<Self, GovernError> {
        Ok(Self::daily(from, to)?.days(Weekday::WORKDAYS))
    }

    /// Sets the days the window is open.
    pub fn days(mut self, days: impl IntoIterator<Item = Weekday>) -> Self {
        self.days = days.into_iter().collect();
        self
    }

    /// Sets the UTC offset from `+HH:MM`, `-HH:MM` or `Z`.
    pub fn utc_offset(mut self, offset: &str) -> Result<Self, GovernError> {
        self.utc_offset_minutes = parse_offset(offset)?;
        Ok(self)
    }

    /// Whether the instant (milliseconds since the Unix epoch) falls in the
    /// window.
    pub fn contains(&self, time_ms: u64) -> bool {
        let local_min = (time_ms / 60_000) as i64 + i64::from(self.utc_offset_minutes);
        let day = local_min.div_euclid(1440);
        let minute = local_min.rem_euclid(1440) as u16;
        let (from, to) = (self.from_minute, self.to_minute);
        let day_ok =
            |d: i64| self.days.is_empty() || self.days.contains(&Weekday::from_days_since_epoch(d));
        if from == to {
            return day_ok(day);
        }
        if from < to {
            return minute >= from && minute < to && day_ok(day);
        }
        // Spans midnight: the evening part belongs to today, the morning
        // part to the day before.
        (minute >= from && day_ok(day)) || (minute < to && day_ok(day - 1))
    }
}

/// Parses `HH:MM` into minutes after midnight (`24:00` is allowed as an
/// end of day).
pub(crate) fn parse_hhmm(text: &str) -> Result<u16, GovernError> {
    let err = || GovernError::Time(format!("expected HH:MM, got `{text}`"));
    let (h, m) = text.trim().split_once(':').ok_or_else(err)?;
    let h: u16 = h.parse().map_err(|_| err())?;
    let m: u16 = m.parse().map_err(|_| err())?;
    if m > 59 || h > 24 || (h == 24 && m != 0) {
        return Err(err());
    }
    Ok((h * 60 + m) % 1440)
}

/// Parses `+HH:MM`, `-HH:MM` or `Z` into minutes.
pub(crate) fn parse_offset(text: &str) -> Result<i16, GovernError> {
    let t = text.trim();
    if t.eq_ignore_ascii_case("z") || t.eq_ignore_ascii_case("utc") {
        return Ok(0);
    }
    let err = || GovernError::Time(format!("expected a UTC offset like -05:00, got `{text}`"));
    let (sign, rest) = match t.as_bytes().first() {
        Some(b'+') => (1, &t[1..]),
        Some(b'-') => (-1, &t[1..]),
        _ => return Err(err()),
    };
    let minutes = parse_hhmm(rest).map_err(|_| err())? as i16;
    if minutes > 14 * 60 {
        return Err(err());
    }
    Ok(sign * minutes)
}

#[cfg(test)]
mod tests {
    use super::*;

    // 2026-11-16T00:00:00Z, a Monday.
    const MONDAY: u64 = 1_794_787_200_000;
    const H: u64 = 3_600_000;

    #[test]
    fn weekday_and_range() {
        let w = TimeWindow::workdays("09:00", "17:00").unwrap();
        assert!(w.contains(MONDAY + 9 * H));
        assert!(w.contains(MONDAY + 17 * H - 60_000));
        assert!(!w.contains(MONDAY + 17 * H));
        assert!(!w.contains(MONDAY + 8 * H));
        assert!(!w.contains(MONDAY + 5 * 24 * H + 10 * H)); // Saturday
        assert!(w.contains(MONDAY + 4 * 24 * H + 10 * H)); // Friday
    }

    #[test]
    fn offset_and_midnight() {
        let w = TimeWindow::workdays("09:00", "17:00")
            .unwrap()
            .utc_offset("-05:00")
            .unwrap();
        assert!(!w.contains(MONDAY + 9 * H)); // 04:00 local
        assert!(w.contains(MONDAY + 14 * H)); // 09:00 local
        let night = TimeWindow::daily("22:00", "06:00")
            .unwrap()
            .days([Weekday::Mon]);
        assert!(night.contains(MONDAY + 23 * H));
        assert!(night.contains(MONDAY + 24 * H + 2 * H)); // Tuesday 02:00, started Monday
        assert!(!night.contains(MONDAY + 2 * H)); // Monday 02:00 started Sunday
        assert!(TimeWindow::daily("00:00", "00:00")
            .unwrap()
            .contains(MONDAY));
    }

    #[test]
    fn parse_errors() {
        assert!(parse_hhmm("9").is_err());
        assert!(parse_hhmm("25:00").is_err());
        assert!(parse_hhmm("10:60").is_err());
        assert_eq!(parse_hhmm("24:00").unwrap(), 0);
        assert_eq!(parse_offset("+05:30").unwrap(), 330);
        assert_eq!(parse_offset("Z").unwrap(), 0);
        assert!(parse_offset("05:00").is_err());
    }
}
