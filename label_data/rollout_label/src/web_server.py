from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, render_template_string, request, send_file
from src.io_utils import CKPT_DIR, PLANS_DIR, RESULTS_DIR, append_jsonl, read_json, utc_ts, write_json
from src.plan import make_plan, save_plan


def generate_seed_from_usr(usr: str) -> int:
    hash_obj = hashlib.sha256(usr.encode("utf-8"))
    seed = int.from_bytes(hash_obj.digest()[:8], byteorder="big")
    return seed % (2**31)


def _reset_progress_if_same_usr(usr: str, plan_id: str):
    results_path = RESULTS_DIR / f"{usr}_{plan_id}.jsonl"
    ckpt_path = CKPT_DIR / f"{usr}_{plan_id}.json"

    removed_any = False
    if results_path.exists():
        results_path.unlink()
        print(f"[fresh] Removed old results: {results_path}")
        removed_any = True
    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"[fresh] Removed old checkpoint: {ckpt_path}")
        removed_any = True
    if not removed_any:
        print("[fresh] No previous results/checkpoint to remove for this plan.")


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rollout Label - {{ usr }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e0e0e0;
        }
        h1 { color: #333; font-size: 24px; margin-bottom: 10px; }
        .user-input-section {
            background: #f9f9f9;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 15px;
            border: 1px solid #e0e0e0;
        }
        .input-row {
            display: flex;
            gap: 15px;
            align-items: center;
            justify-content: center;
            flex-wrap: wrap;
        }
        .input-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        .input-group label {
            font-size: 12px;
            color: #666;
            font-weight: 500;
        }
        .input-group input, .input-group select {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            min-width: 150px;
        }
        .btn-load {
            background: #2196F3;
            color: white;
            padding: 8px 24px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            margin-top: 18px;
            transition: all 0.2s;
        }
        .btn-load:hover { background: #0b7dda; }
        .status {
            color: #666;
            font-size: 16px;
        }
        .file-paths {
            margin-top: 10px;
            padding: 10px;
            background: #f0f8ff;
            border: 1px solid #b3d9ff;
            border-radius: 4px;
            font-size: 13px;
        }
        .path-item {
            margin: 5px 0;
            font-family: monospace;
            word-break: break-all;
        }

        /* Video section */
        .video-section {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        .video-panel {
            flex: 2;
            background: #f9f9f9;
            border-radius: 6px;
            padding: 15px;
            border: 2px solid #e0e0e0;
        }
        .video-panel video {
            width: 100%;
            border-radius: 4px;
            background: #000;
        }
        .video-path {
            margin-top: 8px;
            padding: 6px;
            background: #f0f0f0;
            border-radius: 4px;
            font-size: 12px;
            color: #666;
            word-break: break-all;
        }

        /* Progress bar and timeline */
        .video-progress {
            margin-top: 10px;
            padding: 8px;
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .progress-bar-container {
            position: relative;
            width: 100%;
            height: 12px;
            background: #e0e0e0;
            border-radius: 6px;
            cursor: pointer;
        }
        .progress-bar-container:hover { height: 16px; }
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #45a049);
            border-radius: 6px;
            width: 0%;
            pointer-events: none;
        }
        .scrubber {
            position: absolute;
            top: 50%;
            transform: translate(-50%, -50%);
            width: 18px;
            height: 18px;
            background: white;
            border: 3px solid #4CAF50;
            border-radius: 50%;
            cursor: grab;
            z-index: 5;
            box-shadow: 0 1px 4px rgba(0,0,0,0.3);
            transition: transform 0.1s;
        }
        .scrubber:hover { transform: translate(-50%, -50%) scale(1.2); }
        .scrubber.dragging { cursor: grabbing; transform: translate(-50%, -50%) scale(1.3); }
        .time-display {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #666;
            font-family: monospace;
            margin-top: 4px;
        }
        .playback-controls {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            margin: 10px 0;
        }
        .btn-pause {
            padding: 6px 18px;
            font-size: 14px;
            font-weight: 700;
            border: 2px solid #333;
            background: #fff;
            color: #333;
            border-radius: 5px;
            cursor: pointer;
            min-width: 80px;
        }
        .btn-pause:hover { background: #f0f0f0; }
        .btn-pause.paused { border-color: #4CAF50; color: #4CAF50; }
        .speed-btn {
            padding: 4px 12px;
            font-size: 13px;
            font-weight: 600;
            border: 2px solid #2196F3;
            background: #fff;
            color: #2196F3;
            border-radius: 5px;
            cursor: pointer;
        }
        .speed-btn:hover { background: #e3f2fd; }
        .speed-display {
            font-size: 14px;
            color: #1976D2;
            font-weight: 700;
            min-width: 55px;
            text-align: center;
            padding: 4px 10px;
            background: #e3f2fd;
            border-radius: 4px;
        }

        /* Label panel (right side) */
        .label-panel {
            flex: 1;
            min-width: 320px;
        }
        .label-section {
            background: #f9f9f9;
            border-radius: 6px;
            padding: 15px;
            border: 1px solid #e0e0e0;
            margin-bottom: 15px;
        }
        .label-section h3 {
            font-size: 15px;
            color: #333;
            margin-bottom: 10px;
            padding-bottom: 6px;
            border-bottom: 1px solid #ddd;
        }

        /* Final progress */
        .progress-btn-row {
            display: flex;
            gap: 8px;
            justify-content: center;
        }
        .progress-btn {
            padding: 8px 16px;
            font-size: 15px;
            font-weight: 700;
            border: 2px solid #ddd;
            background: #fff;
            color: #333;
            border-radius: 6px;
            cursor: pointer;
            min-width: 60px;
            transition: all 0.15s;
        }
        .progress-btn:hover { border-color: #4CAF50; background: #f0fff0; }
        .progress-btn.selected {
            border-color: #4CAF50;
            background: #4CAF50;
            color: white;
        }

        /* Segment timeline */
        .segment-timeline {
            position: relative;
            width: 100%;
            height: 40px;
            background: #e8e8e8;
            border-radius: 4px;
            margin: 10px 0;
            cursor: crosshair;
            overflow: hidden;
        }
        .segment-block {
            position: absolute;
            top: 0;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 600;
            color: white;
            border-right: 2px solid white;
            cursor: pointer;
            user-select: none;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            padding: 0 4px;
        }
        .segment-block:last-child { border-right: none; }
        .seg-progress-fast { background: #4CAF50; }
        .seg-progress-slow { background: #8BC34A; }
        .seg-adjust { background: #FF9800; }
        .seg-mistake { background: #f44336; }
        .seg-catastrophic { background: #880E4F; }
        .seg-unlabeled { background: #9E9E9E; }

        .segment-list {
            max-height: 250px;
            overflow-y: auto;
        }
        .segment-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 8px;
            margin-bottom: 4px;
            background: white;
            border-radius: 4px;
            border: 1px solid #e0e0e0;
            font-size: 13px;
        }
        .segment-item.selected {
            border-color: #2196F3;
            box-shadow: 0 0 0 2px rgba(33,150,243,0.3);
        }
        .segment-time {
            font-family: monospace;
            font-size: 12px;
            color: #666;
            min-width: 90px;
        }
        .segment-label-select {
            padding: 3px 6px;
            border: 1px solid #ddd;
            border-radius: 3px;
            font-size: 12px;
            flex: 1;
        }
        .segment-delete {
            background: #f44336;
            color: white;
            border: none;
            border-radius: 3px;
            padding: 2px 8px;
            cursor: pointer;
            font-size: 12px;
        }
        .segment-delete:hover { background: #d32f2f; }

        .segment-instructions {
            font-size: 12px;
            color: #888;
            margin-bottom: 8px;
        }
        .btn-add-split {
            background: #2196F3;
            color: white;
            padding: 6px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .btn-add-split:hover { background: #1976D2; }

        /* Catastrophic failure */
        .catastrophic-section {
            background: #fff3f3;
            border-color: #ffcdd2;
        }
        .btn-catastrophic {
            background: #d32f2f;
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            width: 100%;
        }
        .btn-catastrophic:hover { background: #b71c1c; }
        .btn-catastrophic.active {
            background: #880E4F;
            box-shadow: 0 0 0 3px rgba(136,14,79,0.3);
        }
        .catastrophic-info {
            font-size: 12px;
            color: #d32f2f;
            margin-top: 8px;
        }
        .catastrophic-marker {
            position: absolute;
            top: 0;
            width: 3px;
            height: 100%;
            background: #880E4F;
            z-index: 10;
        }
        .catastrophic-marker::after {
            content: "X";
            position: absolute;
            top: -16px;
            left: -6px;
            font-size: 14px;
            font-weight: bold;
            color: #880E4F;
        }

        /* Navigation */
        .nav-controls {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-top: 15px;
        }
        button {
            padding: 10px 20px;
            font-size: 14px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
            font-weight: 500;
        }
        button:hover {
            transform: translateY(-1px);
            box-shadow: 0 3px 6px rgba(0,0,0,0.15);
        }
        .btn-save {
            background: #4CAF50;
            color: white;
            padding: 10px 32px;
            font-size: 16px;
        }
        .btn-save:hover { background: #388E3C; }
        .btn-nav {
            background: #757575;
            color: white;
        }
        .btn-nav:hover { background: #616161; }

        .message {
            padding: 12px;
            margin: 10px 0;
            border-radius: 6px;
            text-align: center;
            font-weight: 500;
        }
        .message.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .message.info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        .message.complete { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; font-size: 18px; }
        .keyboard-hint {
            text-align: center;
            color: #999;
            font-size: 13px;
            margin-top: 10px;
        }
        .loading { text-align: center; padding: 40px; color: #666; }

        .add-split-hint {
            font-size: 11px;
            color: #999;
            text-align: center;
            margin-top: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Rollout Episode Labeling</h1>
            <div class="user-input-section">
                <div class="input-row">
                    <div class="input-group">
                        <label for="usr-input">User ID</label>
                        <input type="text" id="usr-input" placeholder="Enter user ID" value="{{ usr }}">
                    </div>
                    <div class="input-group">
                        <label for="mode-input">Mode</label>
                        <select id="mode-input">
                            <option value="new">New (create new plan)</option>
                            <option value="resume" selected>Resume (continue existing)</option>
                        </select>
                    </div>
                    <button class="btn-load" onclick="loadPlan()">Load Plan</button>
                </div>
            </div>
            <div class="status">User: <strong id="current-usr">{{ usr }}</strong> | Plan: <strong id="current-plan">{{ plan_id }}</strong> | Episode <strong id="current-idx">-</strong>/<strong id="total-episodes">{{ total }}</strong></div>
            <div class="file-paths" id="file-paths" style="display: none;">
                <div class="path-item"><strong>Results:</strong> <span id="results-path">-</span></div>
                <div class="path-item"><strong>Checkpoint:</strong> <span id="checkpoint-path">-</span></div>
            </div>
        </div>

        <div id="message-area"></div>
        <div id="content-area" class="loading">Loading...</div>
    </div>

    <script>
        let currentIdx = 0;
        let total = {{ total }};
        let currentUsr = '{{ usr }}';
        let currentPlanId = '{{ plan_id }}';
        let videoDuration = 0;

        // Label state for current episode
        let startProgress = 0.4;
        let finalProgress = 1.0;
        const START_PROGRESS_CHOICES = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
        const PROGRESS_CHOICES = [0, 0.1, 0.4, 0.6, 0.9, 1.0];
        let segments = [];  // [{start: 0, end: 0.5, label: "progress_fast"}, ...]
        let catastrophicFrame = null;  // null or time in seconds
        let selectedSegmentIdx = -1;

        const LABELS = [
            {value: "progress_fast", text: "Progress (fast)", cls: "seg-progress-fast"},
            {value: "progress_slow", text: "Progress (slow)", cls: "seg-progress-slow"},
            {value: "adjust", text: "Adjust", cls: "seg-adjust"},
            {value: "mistake", text: "Mistake", cls: "seg-mistake"},
            {value: "catastrophic", text: "Catastrophic failure", cls: "seg-catastrophic"},
        ];

        function labelCls(label) {
            const l = LABELS.find(x => x.value === label);
            return l ? l.cls : "seg-unlabeled";
        }
        function labelText(label) {
            const l = LABELS.find(x => x.value === label);
            return l ? l.text : "Unlabeled";
        }

        window.addEventListener('DOMContentLoaded', () => {
            {% if results_path %}
            document.getElementById('results-path').textContent = '{{ results_path }}';
            document.getElementById('checkpoint-path').textContent = '{{ checkpoint_path }}';
            document.getElementById('file-paths').style.display = 'block';
            {% endif %}
        });

        function showMessage(text, type = 'info') {
            const msgArea = document.getElementById('message-area');
            msgArea.innerHTML = `<div class="message ${type}">${text}</div>`;
            setTimeout(() => { msgArea.innerHTML = ''; }, 3000);
        }

        function formatTime(seconds) {
            if (isNaN(seconds)) return '0:00';
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${mins}:${secs.toString().padStart(2, '0')}`;
        }

        async function loadPlan(confirmed = false) {
            const usr = document.getElementById('usr-input').value.trim();
            const mode = document.getElementById('mode-input').value;
            if (!usr) { showMessage('Please enter a user ID'); return; }

            try {
                document.getElementById('content-area').innerHTML = '<div class="loading">Loading plan...</div>';
                const response = await fetch('/load_plan', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ usr, mode, confirmed })
                });
                const data = await response.json();
                if (data.needs_confirm) {
                    // Show confirmation prompt
                    document.getElementById('content-area').innerHTML = `
                        <div style="text-align:center; padding:40px;">
                            <div style="background:#fff3cd; border:2px solid #ffc107; border-radius:8px; padding:24px; max-width:500px; margin:0 auto;">
                                <h3 style="color:#856404; margin-bottom:12px;">Warning</h3>
                                <p style="color:#856404; margin-bottom:20px;">${data.message}</p>
                                <p style="color:#666; font-size:13px; margin-bottom:16px;">Press <strong>Enter</strong> to confirm and start over, or <strong>Escape</strong> to cancel.</p>
                            </div>
                        </div>
                    `;
                    const handler = (e) => {
                        if (e.key === 'Enter') {
                            document.removeEventListener('keydown', handler);
                            loadPlan(true);
                        } else if (e.key === 'Escape') {
                            document.removeEventListener('keydown', handler);
                            document.getElementById('content-area').innerHTML = '<div class="loading">Cancelled. Choose resume mode or a different user.</div>';
                        }
                    };
                    document.addEventListener('keydown', handler);
                    return;
                }
                if (data.success) {
                    currentUsr = data.usr;
                    currentPlanId = data.plan_id;
                    total = data.total;
                    document.getElementById('current-usr').textContent = data.usr;
                    document.getElementById('current-plan').textContent = data.plan_id;
                    document.getElementById('total-episodes').textContent = data.total;
                    if (data.results_path) {
                        document.getElementById('results-path').textContent = data.results_path;
                        document.getElementById('checkpoint-path').textContent = data.checkpoint_path;
                        document.getElementById('file-paths').style.display = 'block';
                    }
                    showMessage(`Loaded plan for ${data.usr}: ${data.message}`, 'success');
                    await loadEpisode();
                } else {
                    showMessage('Error: ' + data.message);
                    document.getElementById('content-area').innerHTML = '<div class="loading">Error loading plan.</div>';
                }
            } catch (error) {
                showMessage('Error loading plan: ' + error.message);
            }
        }

        function resetLabelState() {
            startProgress = 0.4;
            finalProgress = 1.0;
            segments = [];
            catastrophicFrame = null;
            selectedSegmentIdx = -1;
        }

        async function loadEpisode() {
            try {
                const response = await fetch('/get_episode');
                const data = await response.json();
                currentIdx = data.idx;
                document.getElementById('current-idx').textContent = currentIdx + 1;

                if (data.completed) {
                    document.getElementById('content-area').innerHTML = '<div class="message complete">All episodes completed! Results saved.</div>';
                    return;
                }

                // Load existing label if any
                if (data.existing_label) {
                    startProgress = data.existing_label.start_progress ?? 0.4;
                    finalProgress = data.existing_label.final_progress;
                    segments = data.existing_label.segments || [];
                    catastrophicFrame = data.existing_label.catastrophic_frame;
                } else {
                    resetLabelState();
                }

                renderEpisode(data);
            } catch (error) {
                showMessage('Error loading episode: ' + error.message);
            }
        }

        function renderEpisode(data) {
            document.getElementById('content-area').innerHTML = `
                <div class="video-section">
                    <div class="video-panel">
                        <video id="video-main" autoplay loop muted playsinline>
                            <source src="/video?path=${encodeURIComponent(data.video_path)}" type="video/mp4">
                        </video>
                        <div class="video-progress">
                            <div class="progress-bar-container" id="progress-container">
                                <div class="progress-bar" id="progress-bar"></div>
                                <div class="scrubber" id="scrubber"></div>
                            </div>
                            <div class="time-display">
                                <span id="time-current">0:00</span>
                                <span id="time-duration">0:00</span>
                            </div>
                        </div>
                        <div class="playback-controls">
                            <button class="speed-btn" onclick="changeSpeed(-0.25)">Slower</button>
                            <button class="btn-pause" id="btn-pause" onclick="togglePause()">Pause</button>
                            <span class="speed-display" id="speed-display">2.00x</span>
                            <button class="speed-btn" onclick="changeSpeed(0.25)">Faster</button>
                        </div>
                        <div class="video-path">${data.video_path}</div>
                    </div>

                    <div class="label-panel">
                        <!-- Start Progress -->
                        <div class="label-section" id="start-progress-section">
                            <h3>Start Progress</h3>
                            <div class="progress-btn-row">
                                ${START_PROGRESS_CHOICES.map(v => `<button class="progress-btn ${Math.abs(startProgress - v) < 0.01 ? 'selected' : ''}" onclick="onStartProgressChange(${v})">${Math.round(v * 100)}%</button>`).join('')}
                            </div>
                        </div>

                        <!-- Final Progress -->
                        <div class="label-section" id="progress-section">
                            <h3>Final Progress</h3>
                            <div class="progress-btn-row">
                                ${PROGRESS_CHOICES.map(v => `<button class="progress-btn ${finalProgress === v ? 'selected' : ''}" onclick="onProgressChange(${v})">${Math.round(v * 100)}%</button>`).join('')}
                            </div>
                        </div>

                        <!-- Segment Labels -->
                        <div class="label-section">
                            <h3>Time Segments</h3>
                            <div class="segment-instructions">Pause the video, then click "Split Here" to add a split at the current time. Label each segment below.</div>
                            <button class="btn-add-split" onclick="splitAtCurrentTime()">Split Here (at current time)</button>
                            <div class="segment-timeline" id="segment-timeline"></div>
                            <div class="add-split-hint">Right-click a segment to merge with next</div>
                            <div class="segment-list" id="segment-list"></div>
                        </div>

                        <!-- Catastrophic Failure -->
                        <div class="label-section catastrophic-section">
                            <h3>Catastrophic Failure (optional)</h3>
                            <button class="btn-catastrophic" id="btn-catastrophic" onclick="toggleCatastrophic()">
                                ${catastrophicFrame !== null ? 'Clear Catastrophic Failure' : 'Mark Current Frame as Catastrophic'}
                            </button>
                            <div class="catastrophic-info" id="catastrophic-info" style="display:${catastrophicFrame !== null ? 'block' : 'none'}">
                                Catastrophic failure at ${catastrophicFrame !== null ? formatTime(catastrophicFrame) : '-'}.
                                Final progress set to -100. All segments after this point are "catastrophic failure".
                            </div>
                        </div>

                        <!-- Save + Nav -->
                        <div class="nav-controls">
                            <button class="btn-nav" onclick="navigate('prev')">Prev</button>
                            <button class="btn-save" onclick="saveAndNext()">Save & Next</button>
                            <button class="btn-nav" onclick="navigate('next')">Next (skip)</button>
                        </div>
                        <div class="keyboard-hint">
                            Space (pause/play) | A/D (seek -/+1s) | S (split here) | Ctrl+S (save & next) | Left/Right (prev/next) | [ / ] (speed)
                        </div>
                    </div>
                </div>
            `;

            initVideo();
            renderSegmentTimeline();
            renderSegmentList();
        }

        let isDragging = false;

        function initVideo() {
            const video = document.getElementById('video-main');
            if (!video) return;
            video.playbackRate = 2.0;

            const scrubber = document.getElementById('scrubber');
            const container = document.getElementById('progress-container');

            video.addEventListener('timeupdate', () => {
                if (isDragging) return;
                const pct = (video.currentTime / video.duration) * 100 || 0;
                document.getElementById('progress-bar').style.width = pct + '%';
                scrubber.style.left = pct + '%';
                document.getElementById('time-current').textContent = formatTime(video.currentTime);
            });
            video.addEventListener('loadedmetadata', () => {
                videoDuration = video.duration;
                document.getElementById('time-duration').textContent = formatTime(video.duration);
                video.playbackRate = 2.0;
                if (segments.length === 0) {
                    segments = [{start: 0, end: videoDuration, label: "progress_fast"}];
                    renderSegmentTimeline();
                    renderSegmentList();
                }
            });

            // Click on progress bar to seek
            container.addEventListener('click', (e) => {
                if (isDragging) return;
                const rect = container.getBoundingClientRect();
                const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
                video.currentTime = pct * video.duration;
            });

            // Scrubber drag
            scrubber.addEventListener('mousedown', (e) => {
                e.preventDefault();
                e.stopPropagation();
                isDragging = true;
                scrubber.classList.add('dragging');
                const wasPlaying = !video.paused;
                if (wasPlaying) video.pause();

                const onMove = (e2) => {
                    const rect = container.getBoundingClientRect();
                    const pct = Math.max(0, Math.min(1, (e2.clientX - rect.left) / rect.width));
                    document.getElementById('progress-bar').style.width = (pct * 100) + '%';
                    scrubber.style.left = (pct * 100) + '%';
                    video.currentTime = pct * video.duration;
                    document.getElementById('time-current').textContent = formatTime(video.currentTime);
                };
                const onUp = () => {
                    isDragging = false;
                    scrubber.classList.remove('dragging');
                    document.removeEventListener('mousemove', onMove);
                    document.removeEventListener('mouseup', onUp);
                    if (wasPlaying) video.play();
                };
                document.addEventListener('mousemove', onMove);
                document.addEventListener('mouseup', onUp);
            });
        }

        function togglePause() {
            const video = document.getElementById('video-main');
            const btn = document.getElementById('btn-pause');
            if (!video || !btn) return;
            if (video.paused) {
                video.play();
                btn.textContent = 'Pause';
                btn.classList.remove('paused');
            } else {
                video.pause();
                btn.textContent = 'Play';
                btn.classList.add('paused');
            }
        }

        function seekVideo(seconds) {
            const video = document.getElementById('video-main');
            if (!video) return;
            video.currentTime = Math.max(0, Math.min(video.duration, video.currentTime + seconds));
        }

        function changeSpeed(delta) {
            const video = document.getElementById('video-main');
            if (!video) return;
            let s = Math.max(0.25, Math.min(8.0, video.playbackRate + delta));
            video.playbackRate = s;
            document.getElementById('speed-display').textContent = s.toFixed(2) + 'x';
        }

        function onStartProgressChange(val) {
            startProgress = parseFloat(val);
            document.querySelectorAll('#start-progress-section .progress-btn').forEach(btn => {
                const btnVal = parseFloat(btn.textContent) / 100;
                btn.classList.toggle('selected', Math.abs(btnVal - startProgress) < 0.01);
            });
        }

        function onProgressChange(val) {
            finalProgress = parseFloat(val);
            // Update button states
            document.querySelectorAll('#progress-section .progress-btn').forEach(btn => {
                const btnVal = parseFloat(btn.textContent) / 100;
                btn.classList.toggle('selected', Math.abs(btnVal - finalProgress) < 0.01);
            });
        }

        // --- Segment management ---

        function splitAtCurrentTime() {
            const video = document.getElementById('video-main');
            if (!video || videoDuration <= 0) return;
            const splitTime = video.currentTime;

            for (let i = 0; i < segments.length; i++) {
                const seg = segments[i];
                if (splitTime > seg.start + 0.1 && splitTime < seg.end - 0.1) {
                    const newSeg = {start: splitTime, end: seg.end, label: seg.label};
                    seg.end = splitTime;
                    segments.splice(i + 1, 0, newSeg);
                    break;
                }
            }
            applyCatastrophicToSegments();
            renderSegmentTimeline();
            renderSegmentList();
        }

        function removeSplit(idx) {
            // Merge segment[idx] with segment[idx+1]
            if (idx < 0 || idx >= segments.length - 1) return;
            segments[idx].end = segments[idx + 1].end;
            segments.splice(idx + 1, 1);
            applyCatastrophicToSegments();
            renderSegmentTimeline();
            renderSegmentList();
        }

        function changeSegmentLabel(idx, label) {
            segments[idx].label = label;
            applyCatastrophicToSegments();
            renderSegmentTimeline();
            renderSegmentList();
        }

        function renderSegmentTimeline() {
            const timeline = document.getElementById('segment-timeline');
            if (!timeline || videoDuration <= 0) return;

            let html = '';
            for (let i = 0; i < segments.length; i++) {
                const seg = segments[i];
                const leftPct = (seg.start / videoDuration) * 100;
                const widthPct = ((seg.end - seg.start) / videoDuration) * 100;
                const cls = labelCls(seg.label);
                html += `<div class="segment-block ${cls}" style="left:${leftPct}%;width:${widthPct}%"
                    onclick="event.stopPropagation(); selectSegment(${i})"
                    oncontextmenu="event.preventDefault(); event.stopPropagation(); removeSplit(${i})"
                    title="${labelText(seg.label)}: ${formatTime(seg.start)} - ${formatTime(seg.end)}"
                >${widthPct > 8 ? labelText(seg.label) : ''}</div>`;
            }

            // Catastrophic marker
            if (catastrophicFrame !== null) {
                const markerPct = (catastrophicFrame / videoDuration) * 100;
                html += `<div class="catastrophic-marker" style="left:${markerPct}%"></div>`;
            }

            timeline.innerHTML = html;
        }

        function renderSegmentList() {
            const list = document.getElementById('segment-list');
            if (!list) return;

            let html = '';
            for (let i = 0; i < segments.length; i++) {
                const seg = segments[i];
                const isCatastrophic = seg.label === 'catastrophic';
                html += `<div class="segment-item ${i === selectedSegmentIdx ? 'selected' : ''}" onclick="selectSegment(${i})">
                    <span class="segment-time">${formatTime(seg.start)} - ${formatTime(seg.end)}</span>
                    <select class="segment-label-select" onchange="changeSegmentLabel(${i}, this.value)" ${isCatastrophic && catastrophicFrame !== null && seg.start >= catastrophicFrame ? 'disabled' : ''}>
                        ${LABELS.map(l => `<option value="${l.value}" ${seg.label === l.value ? 'selected' : ''}>${l.text}</option>`).join('')}
                    </select>
                    ${segments.length > 1 && i < segments.length - 1 ? `<button class="segment-delete" onclick="event.stopPropagation(); removeSplit(${i})">Merge</button>` : ''}
                </div>`;
            }
            list.innerHTML = html;
        }

        function selectSegment(idx) {
            selectedSegmentIdx = idx;
            // Seek video to segment start
            const video = document.getElementById('video-main');
            if (video && segments[idx]) {
                video.currentTime = segments[idx].start;
            }
            renderSegmentList();
        }

        // --- Catastrophic failure ---

        function toggleCatastrophic() {
            const video = document.getElementById('video-main');
            if (catastrophicFrame !== null) {
                // Clear
                catastrophicFrame = null;
                segments.forEach(seg => {
                    if (seg.label === 'catastrophic') seg.label = 'progress_fast';
                });
                document.getElementById('btn-catastrophic').textContent = 'Mark Current Frame as Catastrophic';
                document.getElementById('btn-catastrophic').classList.remove('active');
                document.getElementById('catastrophic-info').style.display = 'none';
            } else {
                // Set
                catastrophicFrame = video ? video.currentTime : 0;
                applyCatastrophicToSegments();
                // Default final progress to 0 for catastrophic
                onProgressChange(0);
                document.getElementById('btn-catastrophic').textContent = 'Clear Catastrophic Failure';
                document.getElementById('btn-catastrophic').classList.add('active');
                document.getElementById('catastrophic-info').textContent =
                    `Catastrophic failure at ${formatTime(catastrophicFrame)}. Segments after this point set to "catastrophic failure". Set final progress to the progress achieved before failure.`;
                document.getElementById('catastrophic-info').style.display = 'block';
            }
            renderSegmentTimeline();
            renderSegmentList();
        }

        function applyCatastrophicToSegments() {
            if (catastrophicFrame === null) return;
            for (let seg of segments) {
                // Segments that start at or after catastrophic frame
                if (seg.start >= catastrophicFrame) {
                    seg.label = 'catastrophic';
                }
                // Segment that contains the catastrophic frame: split if needed
                else if (seg.end > catastrophicFrame && seg.start < catastrophicFrame) {
                    // Check if already split at this point
                    const nextIdx = segments.indexOf(seg) + 1;
                    if (nextIdx < segments.length && Math.abs(segments[nextIdx].start - catastrophicFrame) < 0.05) {
                        continue;  // already split
                    }
                    // Split
                    const newSeg = {start: catastrophicFrame, end: seg.end, label: 'catastrophic'};
                    seg.end = catastrophicFrame;
                    segments.splice(segments.indexOf(seg) + 1, 0, newSeg);
                }
            }
        }

        // --- Save & Navigate ---

        async function saveAndNext() {
            const effectiveProgress = finalProgress;
            const label = {
                start_progress: startProgress,
                final_progress: effectiveProgress,
                segments: segments.map(s => ({
                    start: parseFloat(s.start.toFixed(3)),
                    end: parseFloat(s.end.toFixed(3)),
                    label: s.label
                })),
                catastrophic_frame: catastrophicFrame !== null ? parseFloat(catastrophicFrame.toFixed(3)) : null
            };

            try {
                const response = await fetch('/save_label', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ label })
                });
                const data = await response.json();
                if (data.success) {
                    showMessage('Saved!', 'success');
                    resetLabelState();
                    await loadEpisode();
                } else {
                    showMessage('Error saving: ' + data.message);
                }
            } catch (error) {
                showMessage('Error saving: ' + error.message);
            }
        }

        async function navigate(direction) {
            try {
                const response = await fetch('/navigate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ direction })
                });
                const data = await response.json();
                if (data.success) {
                    resetLabelState();
                    await loadEpisode();
                } else {
                    showMessage(data.message || 'Cannot navigate');
                }
            } catch (error) {
                showMessage('Error: ' + error.message);
            }
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
            if (e.ctrlKey && e.key === 's') { e.preventDefault(); saveAndNext(); }
            else if (e.key === ' ') { e.preventDefault(); togglePause(); }
            else if (e.key === 's' || e.key === 'S') splitAtCurrentTime();
            else if (e.key === 'a' || e.key === 'A') { e.preventDefault(); seekVideo(-1); }
            else if (e.key === 'd' || e.key === 'D') { e.preventDefault(); seekVideo(1); }
            else if (e.key === 'ArrowLeft') navigate('prev');
            else if (e.key === 'ArrowRight') navigate('next');
            else if (e.key === '[') changeSpeed(-0.25);
            else if (e.key === ']') changeSpeed(0.25);
        });

        loadEpisode();
    </script>
</body>
</html>
"""


class LabelWebServer:
    def __init__(self, plan_dict: Dict[str, Any], usr: str, host: str, port: int, root: str):
        self.host = host
        self.port = port
        self.root = root

        if plan_dict:
            self._load_plan_from_dict(plan_dict, usr)
        else:
            self.usr = usr
            self.plan = {"plan_id": "none", "episodes": []}
            self.plan_id = "none"
            self.episodes = []
            self.total = 0
            self.results_path = None
            self.ckpt_path = None
            self.idx = 0

        self.app = Flask(__name__)
        self._setup_routes()

    def _load_plan_from_dict(self, plan_dict: Dict[str, Any], usr: str):
        self.usr = usr
        self.plan = plan_dict
        self.plan_id = self.plan["plan_id"]
        self.episodes = self.plan["episodes"]
        self.total = len(self.episodes)
        self.results_path = RESULTS_DIR / f"{usr}_{self.plan_id}.jsonl"
        self.ckpt_path = CKPT_DIR / f"{usr}_{self.plan_id}.json"
        self.idx = 0
        if self.ckpt_path.exists():
            try:
                ck = read_json(self.ckpt_path)
                self.idx = int(ck.get("next_index", 0))
            except Exception:
                self.idx = 0

    def _load_or_create_plan(self, usr: str, mode: str) -> tuple[bool, str]:
        try:
            if mode == "new":
                seed = generate_seed_from_usr(usr)
                plan = make_plan(self.root, seed=seed)
                plan_path = save_plan(plan, usr)
                _reset_progress_if_same_usr(usr, plan.plan_id)
                plan_dict = read_json(plan_path)
                self._load_plan_from_dict(plan_dict, usr)
                return True, f"Created new plan with {self.total} episodes"
            else:
                cand_plans = sorted(PLANS_DIR.glob(f"{usr}_*.json"))
                if not cand_plans:
                    return False, f"No existing plan for user '{usr}'. Use 'new' mode first."
                plan_path = cand_plans[-1]
                plan_dict = read_json(plan_path)
                self._load_plan_from_dict(plan_dict, usr)
                return True, f"Resumed plan with {self.total} episodes (at episode {self.idx + 1})"
        except Exception as e:
            return False, str(e)

    def _get_existing_label(self) -> dict | None:
        """Check if current episode already has a saved label."""
        if not self.results_path or not self.results_path.exists():
            return None
        try:
            with self.results_path.open("r") as f:
                for line in f:
                    row = __import__("json").loads(line)
                    if row.get("episode_index") == self.idx:
                        return row.get("label")
        except Exception:
            pass
        return None

    def _setup_routes(self):
        @self.app.route("/")
        def index():
            results_path = str(self.results_path) if self.results_path else None
            checkpoint_path = str(self.ckpt_path) if self.ckpt_path else None
            return render_template_string(
                HTML_TEMPLATE,
                usr=self.usr,
                plan_id=self.plan_id,
                total=self.total,
                results_path=results_path,
                checkpoint_path=checkpoint_path,
            )

        @self.app.route("/get_episode")
        def get_episode():
            if self.idx >= self.total:
                return jsonify({"completed": True, "idx": self.idx})
            ep = self.episodes[self.idx]
            existing = self._get_existing_label()
            return jsonify({
                "completed": False,
                "idx": self.idx,
                "video_path": ep["video_path"],
                "episode_id": ep["episode_id"],
                "existing_label": existing,
            })

        @self.app.route("/video")
        def serve_video():
            video_path = request.args.get("path", "")
            if not video_path:
                return "No video path provided", 400
            path = Path(video_path)
            if not path.exists():
                return f"Video not found: {video_path}", 404
            mime_type, _ = mimetypes.guess_type(str(path))
            if mime_type is None:
                mime_type = "video/mp4"
            return send_file(str(path), mimetype=mime_type)

        @self.app.route("/save_label", methods=["POST"])
        def save_label():
            data = request.json
            label = data.get("label", {})
            if self.idx >= self.total:
                return jsonify({"success": False, "message": "No more episodes"})

            ep = self.episodes[self.idx]
            row = {
                "ts": utc_ts(),
                "user": self.usr,
                "plan_id": self.plan_id,
                "episode_index": self.idx,
                "episode_id": ep.get("episode_id", self.idx),
                "video_path": ep["video_path"],
                "label": label,
            }
            append_jsonl(self.results_path, row)
            self.idx += 1
            self._save_ckpt()
            return jsonify({"success": True})

        @self.app.route("/navigate", methods=["POST"])
        def navigate():
            data = request.json
            direction = data.get("direction", "")
            if direction == "prev":
                if self.idx > 0:
                    self.idx -= 1
                    return jsonify({"success": True})
                return jsonify({"success": False, "message": "Already at first episode"})
            elif direction == "next":
                if self.idx < self.total - 1:
                    self.idx += 1
                    return jsonify({"success": True})
                return jsonify({"success": False, "message": "Already at last episode"})
            return jsonify({"success": False, "message": "Invalid direction"}), 400

        @self.app.route("/load_plan", methods=["POST"])
        def load_plan():
            data = request.json
            usr = data.get("usr", "").strip()
            mode = data.get("mode", "resume")
            confirmed = data.get("confirmed", False)
            if not usr:
                return jsonify({"success": False, "message": "User ID is required"})

            # Check for existing data when mode is "new"
            if mode == "new" and not confirmed:
                existing_results = sorted(RESULTS_DIR.glob(f"{usr}_*.jsonl"))
                existing_ckpts = sorted(CKPT_DIR.glob(f"{usr}_*.json"))
                has_existing = any(f.stat().st_size > 0 for f in existing_results) or len(existing_ckpts) > 0
                if has_existing:
                    count = 0
                    for f in existing_results:
                        with f.open() as fh:
                            count += sum(1 for _ in fh)
                    return jsonify({
                        "needs_confirm": True,
                        "message": f"User '{usr}' already has {count} labeled episode(s). Starting over will erase all previous progress."
                    })

            success, message = self._load_or_create_plan(usr, mode)
            if success:
                return jsonify({
                    "success": True,
                    "usr": self.usr,
                    "plan_id": self.plan_id,
                    "total": self.total,
                    "message": message,
                    "results_path": str(self.results_path),
                    "checkpoint_path": str(self.ckpt_path),
                })
            return jsonify({"success": False, "message": message})

    def _save_ckpt(self, next_index: int | None = None):
        if next_index is None:
            next_index = self.idx
        write_json(self.ckpt_path, {
            "user": self.usr,
            "plan_id": self.plan_id,
            "next_index": next_index,
        })

    def run(self):
        print("\n" + "=" * 80)
        print("ROLLOUT EPISODE LABELING - WEB SERVER MODE")
        print("=" * 80)
        print(f"User: {self.usr}")
        print(f"Plan ID: {self.plan_id}")
        print(f"Total episodes: {self.total}")
        print(f"Starting at episode: {self.idx + 1}")
        print(f"\nServer starting at: http://{self.host}:{self.port}")
        print(f"Access from your browser at: http://{self.host}:{self.port}")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 80 + "\n")
        self.app.run(host=self.host, port=self.port, debug=False)
