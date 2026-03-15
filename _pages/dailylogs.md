---
title: "DailyLogs"
permalink: /dailylogs/
layout: single
author_profile: false
classes: wide
---

<style>
  .dailylogs-page {
    --dl-bg: #14171b;
    --dl-surface: rgba(32, 35, 42, 0.94);
    --dl-surface-2: #2a2f38;
    --dl-border: rgba(255, 255, 255, 0.08);
    --dl-text: #f2f3f7;
    --dl-muted: #a9afba;
    --dl-accent: #3373c2;
    --dl-accent-soft: #253447;
    --dl-sleep: #7489e0;
    --dl-wake: #e4a840;
    --dl-meal: #3d8f6d;
    --dl-shower: #4aa0c8;
    color: var(--dl-text);
    margin: -1.5rem calc(50% - 50vw) -2rem;
    padding: 0 0 4rem;
    background:
      radial-gradient(circle at top right, rgba(58, 92, 148, 0.32), transparent 26rem),
      radial-gradient(circle at 20% 10%, rgba(90, 128, 196, 0.16), transparent 20rem),
      linear-gradient(180deg, #171b21 0%, #101317 100%);
  }

  .dailylogs-shell {
    max-width: 1120px;
    margin: 0 auto;
    padding: 2.5rem 1.5rem 0;
  }

  .dailylogs-hero {
    display: grid;
    gap: 2rem;
    grid-template-columns: minmax(0, 1.2fr) minmax(300px, 0.8fr);
    align-items: center;
    padding: 2rem 0 3rem;
  }

  .dailylogs-kicker {
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.5rem 0.9rem;
    border: 1px solid var(--dl-border);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.04);
    color: var(--dl-muted);
    font-size: 0.9rem;
    letter-spacing: 0.02em;
  }

  .dailylogs-title {
    margin: 1rem 0 0.8rem;
    font-family: "Avenir Next Rounded", "SF Pro Rounded", ui-rounded, "Trebuchet MS", sans-serif;
    font-size: clamp(2.8rem, 7vw, 5.6rem);
    line-height: 0.96;
    letter-spacing: -0.05em;
    color: var(--dl-text);
  }

  .dailylogs-subtitle {
    max-width: 36rem;
    margin: 0 0 1.4rem;
    font-size: 1.15rem;
    line-height: 1.7;
    color: var(--dl-muted);
  }

  .dailylogs-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.9rem;
    margin-top: 1.5rem;
  }

  .dailylogs-actions a {
    text-decoration: none;
  }

  .dailylogs-primary,
  .dailylogs-secondary {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 11rem;
    padding: 0.95rem 1.25rem;
    border-radius: 1rem;
    font-weight: 700;
    transition: transform 0.2s ease, opacity 0.2s ease, border-color 0.2s ease;
  }

  .dailylogs-primary {
    background: #0c0d10;
    color: #fff;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 14px 28px rgba(0, 0, 0, 0.22);
  }

  .dailylogs-secondary {
    background: rgba(255, 255, 255, 0.06);
    color: var(--dl-text);
    border: 1px solid var(--dl-border);
  }

  .dailylogs-primary:hover,
  .dailylogs-secondary:hover {
    transform: translateY(-2px);
    opacity: 0.96;
  }

  .dailylogs-phone {
    position: relative;
    justify-self: center;
    width: min(100%, 350px);
    padding: 1.25rem;
    border-radius: 2.25rem;
    background: linear-gradient(180deg, rgba(42, 47, 56, 0.82), rgba(24, 27, 33, 0.96));
    border: 1px solid var(--dl-border);
    box-shadow: 0 30px 60px rgba(0, 0, 0, 0.28);
  }

  .dailylogs-phone::before {
    content: "";
    position: absolute;
    top: 0.8rem;
    left: 50%;
    width: 8rem;
    height: 1.6rem;
    border-radius: 999px;
    transform: translateX(-50%);
    background: rgba(0, 0, 0, 0.65);
  }

  .dailylogs-phone-screen {
    overflow: hidden;
    border-radius: 1.65rem;
    background: linear-gradient(180deg, #1a1f28 0%, #151921 100%);
    padding: 3.5rem 1.25rem 1.2rem;
    min-height: 39rem;
  }

  .dailylogs-screen-top {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    margin-bottom: 2.3rem;
  }

  .dailylogs-screen-top img {
    width: 3.25rem;
    height: 3.25rem;
    border-radius: 0.95rem;
    box-shadow: 0 12px 20px rgba(0, 0, 0, 0.18);
  }

  .dailylogs-screen-kicker {
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--dl-muted);
  }

  .dailylogs-screen-headline {
    margin: 0.35rem 0 0;
    font-family: "Avenir Next Rounded", "SF Pro Rounded", ui-rounded, "Trebuchet MS", sans-serif;
    font-size: 2.15rem;
    line-height: 1.05;
    color: var(--dl-text);
  }

  .dailylogs-mini-card {
    margin-top: 1rem;
    padding: 1rem 1rem 0.95rem;
    border-radius: 1.25rem;
    background: rgba(255, 255, 255, 0.045);
    border: 1px solid var(--dl-border);
  }

  .dailylogs-mini-card h3 {
    margin: 0 0 0.35rem;
    font-size: 1rem;
    color: var(--dl-text);
  }

  .dailylogs-mini-card p {
    margin: 0;
    font-size: 0.92rem;
    line-height: 1.55;
    color: var(--dl-muted);
  }

  .dailylogs-grid {
    display: grid;
    gap: 1rem;
    margin-top: 3rem;
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }

  .dailylogs-pill {
    padding: 1.2rem;
    border-radius: 1.35rem;
    border: 1px solid var(--dl-border);
    background: rgba(255, 255, 255, 0.04);
  }

  .dailylogs-pill span {
    display: inline-block;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--dl-muted);
  }

  .dailylogs-pill strong {
    display: block;
    margin-top: 0.65rem;
    font-size: 1.05rem;
    line-height: 1.45;
    color: var(--dl-text);
  }

  .dailylogs-section {
    margin-top: 4.25rem;
  }

  .dailylogs-section-head {
    max-width: 42rem;
    margin-bottom: 1.5rem;
  }

  .dailylogs-section-head h2 {
    margin: 0 0 0.6rem;
    font-family: "Avenir Next Rounded", "SF Pro Rounded", ui-rounded, "Trebuchet MS", sans-serif;
    font-size: clamp(2rem, 4vw, 3rem);
    line-height: 1.02;
    letter-spacing: -0.04em;
    color: var(--dl-text);
  }

  .dailylogs-section-head p {
    margin: 0;
    font-size: 1.05rem;
    line-height: 1.7;
    color: var(--dl-muted);
  }

  .dailylogs-feature-grid {
    display: grid;
    gap: 1rem;
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .dailylogs-feature {
    padding: 1.4rem;
    border-radius: 1.45rem;
    border: 1px solid var(--dl-border);
    background: var(--dl-surface);
    backdrop-filter: blur(10px);
    box-shadow: 0 18px 38px rgba(0, 0, 0, 0.18);
  }

  .dailylogs-feature-mark {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2.25rem;
    height: 2.25rem;
    border-radius: 0.8rem;
    margin-bottom: 0.9rem;
    font-weight: 800;
    color: white;
  }

  .dailylogs-feature h3 {
    margin: 0 0 0.45rem;
    font-size: 1.2rem;
    color: var(--dl-text);
  }

  .dailylogs-feature p {
    margin: 0;
    font-size: 0.98rem;
    line-height: 1.65;
    color: var(--dl-muted);
  }

  .dailylogs-flow {
    display: grid;
    gap: 1rem;
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .dailylogs-step {
    padding: 1.5rem;
    border-radius: 1.45rem;
    border: 1px solid var(--dl-border);
    background: linear-gradient(180deg, rgba(38, 43, 51, 0.94), rgba(27, 30, 36, 0.98));
  }

  .dailylogs-step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2rem;
    height: 2rem;
    margin-bottom: 1rem;
    border-radius: 999px;
    background: var(--dl-accent-soft);
    color: #d9e7fb;
    font-weight: 800;
  }

  .dailylogs-step h3 {
    margin: 0 0 0.45rem;
    color: var(--dl-text);
  }

  .dailylogs-step p {
    margin: 0;
    color: var(--dl-muted);
    line-height: 1.65;
  }

  .dailylogs-footer-card {
    margin-top: 4rem;
    padding: 1.6rem;
    border-radius: 1.6rem;
    border: 1px solid var(--dl-border);
    background: linear-gradient(135deg, rgba(38, 52, 71, 0.98), rgba(24, 28, 35, 0.98));
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    align-items: center;
    justify-content: space-between;
  }

  .dailylogs-footer-card h3 {
    margin: 0 0 0.35rem;
    color: var(--dl-text);
  }

  .dailylogs-footer-card p {
    margin: 0;
    color: rgba(240, 244, 250, 0.78);
  }

  @media (max-width: 960px) {
    .dailylogs-hero,
    .dailylogs-grid,
    .dailylogs-feature-grid,
    .dailylogs-flow {
      grid-template-columns: 1fr;
    }

    .dailylogs-phone {
      width: min(100%, 420px);
    }
  }
</style>

<div class="dailylogs-page">
  <div class="dailylogs-shell">
    <section class="dailylogs-hero">
      <div>
        <div class="dailylogs-kicker">iPhone app · SwiftUI · HealthKit · Firebase</div>
        <h1 class="dailylogs-title">Just a few things each day.</h1>
        <p class="dailylogs-subtitle">
          DailyLogs is a calm, iPhone-first tracker for wake time, sleep, meals, and showers. It keeps the logging flow small on purpose, then turns those tiny entries into trends you can actually use.
        </p>
        <div class="dailylogs-actions">
          <a class="dailylogs-primary" href="https://github.com/bowenyu066/daily-logs">View on GitHub</a>
          <a class="dailylogs-secondary" href="mailto:bowenyu@mit.edu?subject=DailyLogs%20beta">Join the beta</a>
        </div>
      </div>

      <div class="dailylogs-phone">
        <div class="dailylogs-phone-screen">
          <div class="dailylogs-screen-top">
            <img src="/images/projects/dailylogs-icon.png" alt="DailyLogs app icon">
            <div>
              <div class="dailylogs-screen-kicker">DailyLogs</div>
              <p class="dailylogs-screen-headline">Wake. Sleep. Meals. Shower.</p>
            </div>
          </div>

          <div class="dailylogs-mini-card">
            <h3>Today is Mar 15</h3>
            <p>See sunrise and sunset, open a day instantly, and keep the timeline focused on one screen.</p>
          </div>

          <div class="dailylogs-mini-card">
            <h3>Sleep</h3>
            <p>Manual bedtime + wake time, or sync stages from Apple Health and review sleep windows over time.</p>
          </div>

          <div class="dailylogs-mini-card">
            <h3>Meals</h3>
            <p>Quick timestamps, optional photos, and just enough structure to build a real habit without friction.</p>
          </div>

          <div class="dailylogs-mini-card">
            <h3>Analytics</h3>
            <p>Only shown after a real streak of records, so trends feel earned instead of noisy.</p>
          </div>
        </div>
      </div>
    </section>

    <section class="dailylogs-grid">
      <div class="dailylogs-pill">
        <span>Focus</span>
        <strong>No giant health dashboard. Only the recurring signals that matter every day.</strong>
      </div>
      <div class="dailylogs-pill">
        <span>Languages</span>
        <strong>English, Simplified Chinese, or system language, with runtime switching.</strong>
      </div>
      <div class="dailylogs-pill">
        <span>Sync</span>
        <strong>Firebase-backed records, profile data, and meal photos across sessions.</strong>
      </div>
      <div class="dailylogs-pill">
        <span>Privacy</span>
        <strong>Apple Sign In, guest mode for testing, and explicit HealthKit permissions.</strong>
      </div>
    </section>

    <section class="dailylogs-section">
      <div class="dailylogs-section-head">
        <h2>Built around a small routine, not a giant system.</h2>
        <p>
          The product direction is intentionally narrow: if the app asks for too many decisions, it gets abandoned. DailyLogs keeps the surface area small enough that daily use still feels lightweight after the novelty is gone.
        </p>
      </div>

      <div class="dailylogs-feature-grid">
        <article class="dailylogs-feature">
          <div class="dailylogs-feature-mark" style="background: var(--dl-sleep);">S</div>
          <h3>Sleep that works with reality</h3>
          <p>Track bedtime and wake time manually, or sync sleep stages from Apple Health. Trend charts stay readable, selection states are clean, and the analytics page only unlocks after seven consecutive logged days.</p>
        </article>

        <article class="dailylogs-feature">
          <div class="dailylogs-feature-mark" style="background: var(--dl-wake);">W</div>
          <h3>Daylight context without clutter</h3>
          <p>Sunrise and sunset sit right in the daily header when location is available, giving each day a little more context without taking over the interface.</p>
        </article>

        <article class="dailylogs-feature">
          <div class="dailylogs-feature-mark" style="background: var(--dl-meal);">M</div>
          <h3>Meals with optional photos</h3>
          <p>Breakfast, lunch, dinner, plus custom slots if you want them. You can attach a photo, keep a time, and let cloud sync carry the record across sessions.</p>
        </article>

        <article class="dailylogs-feature">
          <div class="dailylogs-feature-mark" style="background: var(--dl-shower);">Q</div>
          <h3>Quiet, fast maintenance habits</h3>
          <p>Showers and other lightweight routines are designed to be logged in seconds. The app favors low-friction check-ins over exhaustive form filling.</p>
        </article>
      </div>
    </section>

    <section class="dailylogs-section">
      <div class="dailylogs-section-head">
        <h2>One screen to log, one screen to learn.</h2>
        <p>
          Home is for action. Analytics is for reflection. Settings is for the few preferences that actually change behavior. That separation keeps the app simple, especially when used every single day.
        </p>
      </div>

      <div class="dailylogs-flow">
        <article class="dailylogs-step">
          <div class="dailylogs-step-number">1</div>
          <h3>Log the day quickly</h3>
          <p>Open today, tap sleep, meals, or shower, and move on. The app favors rounded controls, direct timestamps, and compact editing sheets.</p>
        </article>

        <article class="dailylogs-step">
          <div class="dailylogs-step-number">2</div>
          <h3>Let the app build context</h3>
          <p>HealthKit, localized date formatting, language preferences, sunrise/sunset, and cloud sync all work in the background so the logging flow stays small.</p>
        </article>

        <article class="dailylogs-step">
          <div class="dailylogs-step-number">3</div>
          <h3>Review meaningful trends</h3>
          <p>Once there is enough continuous data, DailyLogs surfaces wake trends, sleep windows, stage durations, meal completion, and other signals without becoming a spreadsheet.</p>
        </article>
      </div>
    </section>

    <section class="dailylogs-footer-card">
      <div>
        <h3>Currently in TestFlight-style beta</h3>
        <p>The app is being tested in real daily use before a wider release. Feedback right now is especially helpful for data sync, clarity, and habit retention.</p>
      </div>
      <div class="dailylogs-actions" style="margin-top: 0;">
        <a class="dailylogs-primary" href="https://github.com/bowenyu066/daily-logs">Source</a>
        <a class="dailylogs-secondary" href="mailto:bowenyu@mit.edu?subject=DailyLogs%20feedback">Send feedback</a>
      </div>
    </section>
  </div>
</div>
