"""Manual Selenium acceptance flow against a running local ThermoTwin server."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as conditions
from selenium.webdriver.support.ui import Select, WebDriverWait


def main(base_url: str = "http://127.0.0.1:5057") -> int:
    options = webdriver.EdgeOptions()
    browser_candidates = [
        shutil.which("msedge"),
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
        Path(os.environ.get("PROGRAMFILES", "")) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
    ]
    browser = next((Path(item) for item in browser_candidates if item and Path(item).is_file()), None)
    if browser is not None:
        options.binary_location = str(browser)
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--force-device-scale-factor=2")
    options.add_argument("--window-size=1440,900")
    options.set_capability("ms:loggingPrefs", {"browser": "ALL"})
    driver = webdriver.Edge(options=options)
    wait = WebDriverWait(driver, 20)
    output = Path("artifacts") / "browser_checks"
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {}
    try:
        driver.get(base_url)
        wait.until(lambda d: "ready" in d.find_element(By.ID, "model-badge").text.lower())
        assert driver.title.startswith("ThermoTwin")
        Select(driver.find_element(By.ID, "scenario")).select_by_value("burst")
        Select(driver.find_element(By.ID, "controller")).select_by_value("hybrid_mpc")
        driver.find_element(By.ID, "new-simulation").click()
        wait.until(lambda d: d.find_element(By.ID, "play").is_enabled())
        driver.find_element(By.ID, "play").click()
        wait.until(lambda d: d.find_element(By.ID, "session-status").get_attribute("data-state") == "running")
        wait.until(lambda d: d.find_element(By.ID, "kpi-clock").text not in {"—", "00:00"})
        driver.find_element(By.ID, "pause").click()
        wait.until(lambda d: d.find_element(By.ID, "session-status").get_attribute("data-state") == "paused")
        result["advanced_clock"] = driver.find_element(By.ID, "kpi-clock").text
        canvas_before = driver.execute_script("const c=document.querySelector('#overview-chart');return {css:c.getBoundingClientRect().height,backing:c.height,page:document.documentElement.scrollHeight}")
        time.sleep(2.3)
        canvas_after = driver.execute_script("const c=document.querySelector('#overview-chart');return {css:c.getBoundingClientRect().height,backing:c.height,page:document.documentElement.scrollHeight}")
        result["overview_canvas_before"] = canvas_before
        result["overview_canvas_after"] = canvas_after
        result["overview_canvas_stable"] = canvas_before == canvas_after and canvas_after["css"] == 260
        driver.find_element(By.CSS_SELECTOR, '[data-tab="forecasts"]').click()
        wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, "#forecast-table tr")) >= 5)
        result["forecast_rows"] = len(driver.find_elements(By.CSS_SELECTOR, "#forecast-table tr"))
        result["forecast_model"] = driver.find_element(By.ID, "forecast-model-state").text
        driver.find_element(By.CSS_SELECTOR, '[data-tab="decisions"]').click()
        wait.until(lambda d: d.find_element(By.ID, "decision-action").text != "No decision yet")
        result["decision_action"] = driver.find_element(By.ID, "decision-action").text
        result["candidate_rows"] = len(driver.find_elements(By.CSS_SELECTOR, "#candidate-table tr"))
        driver.find_element(By.CSS_SELECTOR, '[data-tab="experiments"]').click()
        wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, "#experiment-results .result-card")) == 6)
        result["experiment_cards"] = 6
        result["complete_evidence_selected"] = all("Artifacts from" in card.text for card in driver.find_elements(By.CSS_SELECTOR, "#experiment-results .result-card"))
        research_button = driver.find_element(By.ID, "run-research")
        guarded_before = not research_button.is_enabled()
        driver.find_element(By.ID, "research-confirm").click()
        guarded_after = research_button.is_enabled()
        result["research_launch_guarded"] = guarded_before and guarded_after
        driver.find_element(By.ID, "research-confirm").click()
        driver.save_screenshot(str(output / "experiments_desktop.png"))
        driver.find_element(By.CSS_SELECTOR, '[data-tab="runs"]').click()
        wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, "#run-list .run-card")) >= 1)
        result["run_cards"] = len(driver.find_elements(By.CSS_SELECTOR, "#run-list .run-card"))
        driver.find_element(By.CSS_SELECTOR, '[data-tab="overview"]').click()
        driver.find_element(By.ID, "reset").click()
        wait.until(lambda d: d.find_element(By.ID, "kpi-clock").text == "00:00")
        driver.save_screenshot(str(output / "dashboard_desktop.png"))
        driver.set_window_size(390, 844)
        driver.find_element(By.CSS_SELECTOR, '[data-tab="thermal"]').click()
        time.sleep(0.5)
        overflow = driver.execute_script("return document.documentElement.scrollWidth - document.documentElement.clientWidth")
        result["mobile_horizontal_overflow_px"] = overflow
        if overflow > 1:
            offenders = driver.execute_script("return [...document.querySelectorAll('*')].filter(e => e.getBoundingClientRect().right > document.documentElement.clientWidth + 1).map(e => ({tag:e.tagName,id:e.id,cls:e.className,right:Math.round(e.getBoundingClientRect().right),width:Math.round(e.getBoundingClientRect().width)})).slice(0,20)")
            result["mobile_overflow_elements"] = offenders
        driver.save_screenshot(str(output / "dashboard_mobile.png"))
        severe = [entry for entry in driver.get_log("browser") if entry.get("level") == "SEVERE"]
        result["severe_console_errors"] = severe
        result["passed"] = overflow <= 1 and not severe and result["overview_canvas_stable"] and result["complete_evidence_selected"] and result["research_launch_guarded"]
    finally:
        driver.quit()
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5057"))
