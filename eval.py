#!/usr/bin/env python3
"""
Druryism Model Evaluator
Author: Principal AI Software Engineer

This script evaluates and compares the performance of the fine-tuned Peter Drury
commentary model (with QLoRA adapters) against the baseline SLM (without adapters).
It runs inference on a 10-scene evaluation dataset (including validation scenes and
out-of-domain generalization scenes), calculates metrics (length, generation speed,
Druryism vocabulary density, dramatic punctuation, vocabulary richness), outputs
detailed data to JSON, and generates a self-contained HTML visual report.
"""

import os
import sys
import time
import json
import gc
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DruryEval")

# Import MLX components dynamically to check for availability
try:
    import mlx.core as mx
except ImportError:
    logger.error("MLX is not installed. Please run on a macOS environment with MLX.")
    sys.exit(1)

# Ensure current directory is in search path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# Import pipeline components
try:
    from pipeline import DruryCommentaryEngine, CommentaryConfig
except ImportError as e:
    logger.critical(f"Failed to import pipeline components from 'pipeline.py'. Error: {e}")
    sys.exit(1)

# -------------------------------------------------------------------------
# Evaluation Dataset Definition
# -------------------------------------------------------------------------
EVAL_SCENES = [
    # 6 Validation Scenes (from data/valid.jsonl)
    {
        "id": 1,
        "type": "Validation",
        "scene": "A veteran player, playing their final match before retirement, gets substituted off. The entire stadium, including opposing fans, gives a standing ovation.",
        "ground_truth": "The curtain falls on an immortal. A standing ovation that transcends tribal loyalty. They applaud the man, the myth, the decades of devotion written in the lines of his face. The game stands still to say goodbye."
    },
    {
        "id": 2,
        "type": "Validation",
        "scene": "A dramatic, multi-player brawl breaks out near the touchline during a highly tense, aggressive local derby match.",
        "ground_truth": "Double, double, toil and trouble! Fires raging on the frontier! The local derby boils over into raw, tribal fury. A boiling cauldron of animosity where tactical shapes dissolve into ancient, visceral combat. This isn't just a match; it's local civil war on grass!"
    },
    {
        "id": 3,
        "type": "Validation",
        "scene": "Luka Modrić delivers an incredible, outside-of-the-boot cross that perfectly bypasses three defenders to assist a crucial goal.",
        "ground_truth": "Modrić, with the trivela... oh, exquisite! A pass of architectural genius, carved out with the outside of his boot. He doth bend reality to his artistic whim. The little magician of Zadar does it again!"
    },
    {
        "id": 4,
        "type": "Validation",
        "scene": "An aggregate-winning penalty kick hits the crossbar, bounces down onto the line, and spins out. The opposing team celebrates their survival.",
        "ground_truth": "Agony and ecstasy separated by a millimeter of aluminum! It rattled the heavens, hit the earth, and refused to cross the threshold! Deliverance for one, damnation for the other. A cruel, beautiful game of fractions! The course of true love never did run smooth!"
    },
    {
        "id": 5,
        "type": "Validation",
        "scene": "A player receiving a standing ovation from opposing fans after scoring a hat-trick against them.",
        "ground_truth": "The stadium rises to applaud the destroyer of their dreams! A hat-trick of such sublime brilliance that tribal walls fall. They pay homage to greatness, for he is a man of men, beyond compare!"
    },
    {
        "id": 6,
        "type": "Validation",
        "scene": "A player hitting a shot that hits both posts before spinning out, leaving everyone in shock.",
        "ground_truth": "Off the post... and off the other post! Oh, what a cruel trick of fortune! It danced on the line, teased the net, and refused to enter. Stunned, breathless silence, as if the fates themselves intervened!"
    },
    # 4 Generalization Scenes (New, out-of-domain)
    {
        "id": 7,
        "type": "Generalization",
        "scene": "A young, inexperienced goalkeeper slips while trying to clear the ball in a cup final, letting the ball roll in for a shocking self-inflicted goal.",
        "ground_truth": "A tragedy in green! The ground swallows him whole, a cruel slip of fate in the grandest theatre of them all. Cruel, unsparing drama! A young boy's heart breaks on the turf as the stadium gasps in horror."
    },
    {
        "id": 8,
        "type": "Generalization",
        "scene": "A player executing a dazzling solo run from his own half, nutmegging two defenders in the penalty area before chipping the keeper.",
        "ground_truth": "He runs like the wind, weaving through a maze of defenders! A solo run of breathtaking audacity! A nutmeg, a chip, a slice of absolute genius! They stood like statues, helpless witnesses to a masterpiece."
    },
    {
        "id": 9,
        "type": "Generalization",
        "scene": "In a torrential downpour, a team scores a scrappy, mud-soaked equalizer in the 95th minute to draw the match.",
        "ground_truth": "Out of the mud and the rain, salvation! A scrappy, rain-drenched scramble that sends the stadium into a frenzy! They fought in the tempest and found their equalizer in the dying, desperate seconds!"
    },
    {
        "id": 10,
        "type": "Generalization",
        "scene": "An underdog team, playing with 9 men, holds off a barrage of attacks from a powerhouse club for 30 minutes to secure a historic draw.",
        "ground_truth": "An unyielding siege, repelled by absolute defiance! Nine men standing like Sparta against a Persian sea of attacks. They have held the line, survived the storm, and written their names in history!"
    }
]

# Tracked "Druryisms" vocabulary keywords
DRURYISMS = {
    "immortal", "bedlam", "destiny", "colossus", "heavens", "fates", "agony", "ecstasy", 
    "miracle", "epic", "fury", "greatness", "coliseum", "cauldron", "theatre", "audacity", 
    "sculptor", "canvas", "titan", "majestic", "drama", "immortality", "curtain", 
    "deliverance", "damnation", "exquisite", "magician", "myth", "tempest", "pantheon", 
    "gods", "poetry", "poetic", "glory", "despair", "saga", "breathless", "unyielding", 
    "defying", "physics", "immense", "battlefield", "tribal", "loyalty", "animosity",
    "combat", "trivela", "genius", "whim", "aluminum", "rattled", "threshold",
    "fractions", "destroyer", "sublime", "homage", "intervened", "tragedy", "salvation",
    "frenzy", "siege", "defiance", "doil", "trouble", "bordering", "razor", "whistle"
}

# -------------------------------------------------------------------------
# Metric Computations
# -------------------------------------------------------------------------
def calculate_metrics(text: str, generation_time: float, token_count: int) -> Dict[str, Any]:
    # Normalize text to lowercase words without punctuation
    words = [w.strip(".,!?\"'()[]:;").lower() for w in text.split()]
    words = [w for w in words if w]
    
    unique_words = set(words)
    word_count = len(words)
    char_count = len(text)
    
    # Druryism matching
    drury_words = [w for w in words if w in DRURYISMS]
    drury_count = len(drury_words)
    drury_density = (drury_count / max(1, word_count)) * 100
    
    # Punctuation counts
    exclamation_count = text.count("!")
    question_count = text.count("?")
    
    # Type-Token Ratio (vocabulary richness)
    ttr = len(unique_words) / max(1, word_count) if word_count > 0 else 0
    
    # Speed (tokens per second)
    speed = token_count / max(0.001, generation_time)
    
    return {
        "text": text,
        "word_count": word_count,
        "char_count": char_count,
        "drury_count": drury_count,
        "drury_words": list(set(drury_words)),
        "drury_density": round(drury_density, 2),
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "ttr": round(ttr, 3),
        "generation_time": round(generation_time, 3),
        "token_count": token_count,
        "speed": round(speed, 2)
    }

# -------------------------------------------------------------------------
# Model Inference Runners
# -------------------------------------------------------------------------
def run_evaluation(config: CommentaryConfig) -> List[Dict[str, Any]]:
    """Runs generation on all evaluation scenes using the given config."""
    engine = DruryCommentaryEngine(config)
    engine.load_model()
    
    results = []
    for item in EVAL_SCENES:
        scene_id = item["id"]
        scene_text = item["scene"]
        scene_type = item["type"]
        
        logger.info(f"Running inference on Scene {scene_id} [{scene_type}]...")
        
        start_time = time.time()
        
        generated_text = ""
        stream = engine.generate_commentary_stream(scene_text)
        for token in stream:
            generated_text += token
            
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Get precise token count using tokenizer
        tokens = engine.tokenizer.encode(generated_text)
        token_count = len(tokens)
        
        metrics = calculate_metrics(generated_text, generation_time, token_count)
        
        results.append({
            "id": scene_id,
            "type": scene_type,
            "scene": scene_text,
            "ground_truth": item.get("ground_truth", ""),
            "metrics": metrics
        })
        
        logger.info(f"Completed scene {scene_id}: {metrics['word_count']} words generated in {metrics['generation_time']}s ({metrics['speed']} tok/s).")
    
    # Free MLX memory
    logger.info("Cleaning up engine memory...")
    del engine.model
    del engine.tokenizer
    gc.collect()
    try:
        mx.metal.clear_cache()
    except Exception as e:
        logger.debug(f"Failed to clear metal cache: {e}")
        
    return results

# -------------------------------------------------------------------------
# Summary Metrics Compiler
# -------------------------------------------------------------------------
def compile_summary(baseline_results: List[Dict[str, Any]], finetuned_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    def get_avg(results, key):
        return sum(item["metrics"][key] for item in results) / len(results)
    
    # Calculate all Druryisms generated across datasets
    def get_all_druryisms(results):
        all_words = set()
        for r in results:
            all_words.update(r["metrics"]["drury_words"])
        return list(all_words)

    return {
        "baseline": {
            "avg_speed": round(get_avg(baseline_results, "speed"), 1),
            "avg_word_count": round(get_avg(baseline_results, "word_count"), 1),
            "avg_drury_count": round(get_avg(baseline_results, "drury_count"), 1),
            "avg_drury_density": round(get_avg(baseline_results, "drury_density"), 2),
            "avg_ttr": round(get_avg(baseline_results, "ttr"), 3),
            "avg_exclamation_count": round(get_avg(baseline_results, "exclamation_count"), 1),
            "avg_generation_time": round(get_avg(baseline_results, "generation_time"), 2),
            "unique_druryisms_hit": len(get_all_druryisms(baseline_results)),
            "all_druryisms_hit": get_all_druryisms(baseline_results)
        },
        "finetuned": {
            "avg_speed": round(get_avg(finetuned_results, "speed"), 1),
            "avg_word_count": round(get_avg(finetuned_results, "word_count"), 1),
            "avg_drury_count": round(get_avg(finetuned_results, "drury_count"), 1),
            "avg_drury_density": round(get_avg(finetuned_results, "drury_density"), 2),
            "avg_ttr": round(get_avg(finetuned_results, "ttr"), 3),
            "avg_exclamation_count": round(get_avg(finetuned_results, "exclamation_count"), 1),
            "avg_generation_time": round(get_avg(finetuned_results, "generation_time"), 2),
            "unique_druryisms_hit": len(get_all_druryisms(finetuned_results)),
            "all_druryisms_hit": get_all_druryisms(finetuned_results)
        }
    }

# -------------------------------------------------------------------------
# HTML Report Generator
# -------------------------------------------------------------------------
def generate_html_report(eval_data: Dict[str, Any], output_path: str):
    """Generates a premium, highly aesthetic HTML visual report."""
    results_json = json.dumps(eval_data, indent=2)
    druryisms_list = list(DRURYISMS)
    druryisms_list.sort()
    druryisms_json = json.dumps(druryisms_list)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Druryism SLM Evaluation Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: #020617;
            --bg-obsidian: #0b1329;
            --bg-card: rgba(15, 23, 42, 0.65);
            --border-card: rgba(255, 255, 255, 0.07);
            --border-card-hover: rgba(99, 102, 241, 0.3);
            --text-main: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.15);
            --secondary: #a855f7;
            --secondary-glow: rgba(168, 85, 247, 0.15);
            --success: #10b981;
            --success-glow: rgba(16, 185, 129, 0.12);
            --warning: #f59e0b;
            --accent-pink: #ec4899;
            --font-mono: 'JetBrains Mono', monospace;
            --font-sans: 'Outfit', sans-serif;
            --transition-smooth: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--bg-dark);
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.12) 0px, transparent 55%),
                radial-gradient(at 50% 0%, rgba(168, 85, 247, 0.12) 0px, transparent 55%),
                radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.05) 0px, transparent 40%);
            color: var(--text-main);
            font-family: var(--font-sans);
            padding: 2.5rem;
            min-height: 100vh;
            line-height: 1.6;
        }}

        header {{
            max-width: 1400px;
            margin: 0 auto 3rem auto;
            text-align: center;
            position: relative;
        }}

        .title-container {{
            display: inline-block;
            position: relative;
        }}

        h1 {{
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            background: linear-gradient(135deg, #a5b4fc 10%, #c084fc 50%, #f472b6 90%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            text-shadow: 0 0 50px rgba(99, 102, 241, 0.25);
        }}

        .subtitle {{
            font-size: 1.15rem;
            color: var(--text-secondary);
            font-weight: 400;
            letter-spacing: 0.01em;
            max-width: 700px;
            margin: 0 auto;
        }}

        .badge-container {{
            margin-top: 1.25rem;
            display: flex;
            justify-content: center;
            gap: 0.75rem;
        }}

        .badge {{
            padding: 0.4rem 0.9rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: var(--text-secondary);
        }}

        .badge.active {{
            background: rgba(99, 102, 241, 0.15);
            border-color: rgba(99, 102, 241, 0.3);
            color: #a5b4fc;
            box-shadow: 0 0 12px rgba(99, 102, 241, 0.1);
        }}

        main {{
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 2.5rem;
        }}

        /* Metrics Cards Grid */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
        }}

        .metric-card {{
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border-card);
            border-radius: 1.25rem;
            padding: 1.75rem;
            position: relative;
            overflow: hidden;
            transition: var(--transition-smooth);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}

        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.03) 0%, transparent 100%);
            pointer-events: none;
        }}

        .metric-card:hover {{
            transform: translateY(-4px);
            border-color: var(--border-card-hover);
            box-shadow: 0 12px 30px rgba(99, 102, 241, 0.08);
        }}

        .metric-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
        }}

        .metric-title {{
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            font-weight: 600;
        }}

        .metric-icon {{
            font-size: 1.5rem;
            opacity: 0.8;
        }}

        .metric-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            align-items: flex-end;
        }}

        .metric-value-box {{
            display: flex;
            flex-direction: column;
        }}

        .model-name {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.02em;
            margin-bottom: 0.25rem;
        }}

        .metric-value {{
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            line-height: 1;
        }}

        .baseline-val {{
            color: var(--text-secondary);
        }}

        .finetuned-val {{
            background: linear-gradient(135deg, #a5b4fc, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .percentage-badge {{
            grid-column: span 2;
            margin-top: 1rem;
            padding: 0.5rem 0.75rem;
            border-radius: 0.75rem;
            font-size: 0.85rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.35rem;
            border: 1px solid transparent;
        }}

        .percentage-badge.positive {{
            background: var(--success-glow);
            border-color: rgba(16, 185, 129, 0.25);
            color: #34d399;
        }}

        .percentage-badge.neutral {{
            background: rgba(255, 255, 255, 0.03);
            border-color: rgba(255, 255, 255, 0.08);
            color: var(--text-secondary);
        }}

        /* Visualization Section */
        .dashboard-row {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 2rem;
        }}

        @media (max-width: 1024px) {{
            .dashboard-row {{
                grid-template-columns: 1fr;
            }}
        }}

        .panel {{
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border-card);
            border-radius: 1.5rem;
            padding: 2rem;
            display: flex;
            flex-direction: column;
        }}

        .panel-title {{
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            color: var(--text-main);
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 0.75rem;
        }}

        .chart-container {{
            width: 100%;
            height: 350px;
            position: relative;
        }}

        svg.chart-svg {{
            width: 100%;
            height: 100%;
        }}

        /* Side-by-Side playground */
        .playground-grid {{
            display: grid;
            grid-template-columns: 320px 1fr;
            gap: 2rem;
            min-height: 600px;
        }}

        @media (max-width: 768px) {{
            .playground-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .scene-list {{
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            overflow-y: auto;
            max-height: 650px;
            padding-right: 0.5rem;
        }}

        .scene-list::-webkit-scrollbar {{
            width: 6px;
        }}

        .scene-list::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 99px;
        }}

        .scene-btn {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 0.85rem;
            padding: 1.15rem;
            text-align: left;
            cursor: pointer;
            transition: var(--transition-smooth);
            color: var(--text-secondary);
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}

        .scene-btn:hover {{
            background: rgba(255, 255, 255, 0.05);
            border-color: rgba(255, 255, 255, 0.12);
        }}

        .scene-btn.active {{
            background: rgba(99, 102, 241, 0.08);
            border-color: rgba(99, 102, 241, 0.35);
            color: var(--text-main);
            box-shadow: inset 0 0 12px rgba(99, 102, 241, 0.05);
        }}

        .scene-badge-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .scene-id-lbl {{
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .scene-type-badge {{
            font-size: 0.65rem;
            font-weight: 700;
            padding: 0.15rem 0.45rem;
            border-radius: 99px;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }}

        .type-validation {{
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.2);
            color: #34d399;
        }}

        .type-generalization {{
            background: rgba(245, 158, 11, 0.1);
            border: 1px solid rgba(245, 158, 11, 0.2);
            color: #fbbf24;
        }}

        .scene-snippet {{
            font-size: 0.85rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            width: 100%;
        }}

        .comparison-details {{
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            animation: fadeIn 0.3s ease;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(8px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .factual-input-box {{
            background: rgba(255, 255, 255, 0.015);
            border: 1px dashed rgba(255, 255, 255, 0.1);
            border-radius: 1rem;
            padding: 1.25rem 1.5rem;
            position: relative;
        }}

        .factual-title {{
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            margin-bottom: 0.35rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .factual-text {{
            font-size: 1.05rem;
            color: var(--text-main);
            font-weight: 400;
        }}

        /* Outputs Side-by-Side */
        .outputs-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }}

        @media (max-width: 900px) {{
            .outputs-container {{
                grid-template-columns: 1fr;
            }}
        }}

        .output-box {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 1.25rem;
            padding: 1.75rem;
            display: flex;
            flex-direction: column;
            position: relative;
        }}

        .output-box.finetuned-box {{
            background: radial-gradient(circle at 100% 0%, rgba(168, 85, 247, 0.05) 0%, rgba(255, 255, 255, 0.02) 70%);
            border-color: rgba(168, 85, 247, 0.25);
            box-shadow: 0 4px 24px rgba(168, 85, 247, 0.03);
        }}

        .output-box.finetuned-box::after {{
            content: '';
            position: absolute;
            top: -1px;
            right: 1.5rem;
            width: 80px;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--secondary), transparent);
        }}

        .output-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.25rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            padding-bottom: 0.75rem;
        }}

        .output-label {{
            font-size: 0.95rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .lbl-baseline {{
            color: var(--text-secondary);
        }}

        .lbl-finetuned {{
            color: #c084fc;
        }}

        .text-content {{
            font-size: 1.15rem;
            line-height: 1.7;
            color: var(--text-main);
            min-height: 140px;
            margin-bottom: 1.5rem;
            letter-spacing: 0.01em;
        }}

        .text-content.poetic-font {{
            font-family: var(--font-sans);
            font-style: italic;
        }}

        /* Highlights */
        .drury-highlight {{
            background: rgba(168, 85, 247, 0.15);
            border-bottom: 1.5px solid rgba(168, 85, 247, 0.5);
            padding: 0.05rem 0.25rem;
            border-radius: 0.25rem;
            font-weight: 600;
            color: #d8b4fe;
            text-shadow: 0 0 8px rgba(168, 85, 247, 0.3);
            display: inline-block;
            margin: 0 0.05rem;
        }}

        .ground-truth-box {{
            border: 1px dashed rgba(16, 185, 129, 0.3);
            background: rgba(16, 185, 129, 0.015);
            border-radius: 1.25rem;
            padding: 1.5rem 1.75rem;
            position: relative;
        }}

        .lbl-gt {{
            color: var(--success);
        }}

        .gt-text {{
            font-size: 1.1rem;
            color: var(--text-secondary);
            font-style: italic;
        }}

        /* Scene Specific Metrics Table */
        .scene-metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}

        .scene-metrics-table th {{
            text-align: left;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}

        .scene-metrics-table td {{
            padding: 0.75rem 0;
            font-size: 0.9rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
        }}

        .scene-metrics-table td.m-val {{
            font-family: var(--font-mono);
            font-weight: 500;
        }}

        /* Druryism Vocabulary Grid */
        .vocab-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
            gap: 0.75rem;
            margin-top: 1.5rem;
        }}

        .vocab-item {{
            padding: 0.6rem;
            border-radius: 0.75rem;
            font-size: 0.85rem;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.03);
            background: rgba(255, 255, 255, 0.01);
            color: var(--text-muted);
            transition: var(--transition-smooth);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.35rem;
        }}

        .vocab-item.hit {{
            background: rgba(168, 85, 247, 0.08);
            border-color: rgba(168, 85, 247, 0.25);
            color: #d8b4fe;
            font-weight: 500;
            box-shadow: 0 0 10px rgba(168, 85, 247, 0.05);
        }}

        .check-icon {{
            font-size: 0.75rem;
            opacity: 0.8;
            color: #a855f7;
        }}

        /* Responsive adjustments */
        @media (max-width: 600px) {{
            body {{
                padding: 1rem;
            }}
            h1 {{
                font-size: 2.2rem;
            }}
            .subtitle {{
                font-size: 1rem;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <div class="title-container">
            <h1>Druryism Commentary Engine</h1>
        </div>
        <p class="subtitle">Quantitative and qualitative evaluation of the local fine-tuned SLM (base Phi-3-mini + QLoRA adapters) vs the baseline model.</p>
        <div class="badge-container">
            <span class="badge active">Local Apple Silicon</span>
            <span class="badge active">Phi-3-Mini-4K (4-bit)</span>
            <span class="badge active">QLoRA Adapters (200 Iters)</span>
        </div>
    </header>

    <main>
        <!-- Metrics Summary Cards -->
        <section class="metrics-grid">
            <!-- Metric 1: Druryisms Count -->
            <div class="metric-card">
                <div class="metric-header">
                    <span class="metric-title">Avg Druryisms Per Scene</span>
                    <span class="metric-icon">✨</span>
                </div>
                <div class="metric-comparison">
                    <div class="metric-value-box">
                        <span class="model-name">Baseline</span>
                        <span class="metric-value baseline-val" id="sum-base-drury">0.0</span>
                    </div>
                    <div class="metric-value-box">
                        <span class="model-name">Fine-Tuned</span>
                        <span class="metric-value finetuned-val" id="sum-ft-drury">0.0</span>
                    </div>
                    <div class="percentage-badge positive" id="sum-drury-pct">
                        +0% Increase in Poetic Imagery
                    </div>
                </div>
            </div>

            <!-- Metric 2: Exclamations -->
            <div class="metric-card">
                <div class="metric-header">
                    <span class="metric-title">Dramatic Delivery (Exclamations)</span>
                    <span class="metric-icon">🔥</span>
                </div>
                <div class="metric-comparison">
                    <div class="metric-value-box">
                        <span class="model-name">Baseline</span>
                        <span class="metric-value baseline-val" id="sum-base-excl">0.0</span>
                    </div>
                    <div class="metric-value-box">
                        <span class="model-name">Fine-Tuned</span>
                        <span class="metric-value finetuned-val" id="sum-ft-excl">0.0</span>
                    </div>
                    <div class="percentage-badge positive" id="sum-excl-pct">
                        +0% Dramatic Intensity
                    </div>
                </div>
            </div>

            <!-- Metric 3: Vocabulary Richness (TTR) -->
            <div class="metric-card">
                <div class="metric-header">
                    <span class="metric-title">Vocabulary Diversity (TTR)</span>
                    <span class="metric-icon">📚</span>
                </div>
                <div class="metric-comparison">
                    <div class="metric-value-box">
                        <span class="model-name">Baseline</span>
                        <span class="metric-value baseline-val" id="sum-base-ttr">0.000</span>
                    </div>
                    <div class="metric-value-box">
                        <span class="model-name">Fine-Tuned</span>
                        <span class="metric-value finetuned-val" id="sum-ft-ttr">0.000</span>
                    </div>
                    <div class="percentage-badge neutral" id="sum-ttr-pct">
                        TTR (Unique Words / Total Words)
                    </div>
                </div>
            </div>

            <!-- Metric 4: Generation Speed -->
            <div class="metric-card">
                <div class="metric-header">
                    <span class="metric-title">Avg Generation Speed</span>
                    <span class="metric-icon">⚡</span>
                </div>
                <div class="metric-comparison">
                    <div class="metric-value-box">
                        <span class="model-name">Baseline</span>
                        <span class="metric-value baseline-val" id="sum-base-speed">0.0 t/s</span>
                    </div>
                    <div class="metric-value-box">
                        <span class="model-name">Fine-Tuned</span>
                        <span class="metric-value finetuned-val" id="sum-ft-speed">0.0 t/s</span>
                    </div>
                    <div class="percentage-badge neutral" id="sum-speed-pct">
                        0.0s vs 0.0s Avg Duration
                    </div>
                </div>
            </div>
        </section>

        <!-- Visualization Row -->
        <section class="dashboard-row">
            <div class="panel">
                <div class="panel-title">
                    <span>Poetic Word count (Druryisms) per Scene</span>
                    <span style="font-size: 0.8rem; font-weight: normal; color: var(--text-secondary);">Baseline (Grey) vs. Fine-Tuned (Indigo)</span>
                </div>
                <div class="chart-container" id="bar-chart-container">
                    <svg class="chart-svg" id="metrics-chart" viewBox="0 0 800 320">
                        <!-- Chart will be drawn dynamically via JS -->
                    </svg>
                </div>
            </div>

            <div class="panel">
                <div class="panel-title">
                    <span>Inference Speed Comparison</span>
                    <span style="font-size: 0.8rem; font-weight: normal; color: var(--text-secondary);">Tokens / Second</span>
                </div>
                <div class="chart-container">
                    <svg class="chart-svg" id="speed-chart" viewBox="0 0 350 320">
                        <!-- Speed chart drawn dynamically -->
                    </svg>
                </div>
            </div>
        </section>

        <!-- Interactive Playground -->
        <section class="panel">
            <div class="panel-title">
                <span>Side-by-Side Commentary Playground</span>
                <span class="badge" style="background: rgba(99, 102, 241, 0.05); border-color: rgba(99, 102, 241, 0.2); color: var(--primary);">Interactive</span>
            </div>

            <div class="playground-grid">
                <!-- Scene Selector List -->
                <div class="scene-list" id="scene-selector-list">
                    <!-- Populated via JS -->
                </div>

                <!-- Detailed Comparison View -->
                <div class="comparison-details" id="comparison-details-view">
                    <!-- Populated via JS -->
                </div>
            </div>
        </section>

        <!-- Druryism Lexicon Coverage -->
        <section class="panel">
            <div class="panel-title">
                <span>Druryism Dictionary Coverage</span>
                <span style="font-size: 0.8rem; font-weight: normal; color: var(--text-secondary);" id="vocab-coverage-ratio">0/0 words hit (0%)</span>
            </div>
            <p style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 1rem; max-width: 900px;">
                The QLoRA adapters adapt the model's token distribution, increasing the likelihood of specific dramatic words. Below is the list of target poetic vocabulary terms tracked during evaluation. Highlighted words were successfully generated in at least one scene by the fine-tuned model.
            </p>
            <div class="vocab-grid" id="vocab-grid-container">
                <!-- Populated via JS -->
            </div>
        </section>
    </main>

    <script>
        // Injecting the raw JSON evaluation data
        const data = {results_json};
        const trackedDruryisms = {druryisms_json};

        // Cache DOM elements
        const sumBaseDrury = document.getElementById('sum-base-drury');
        const sumFtDrury = document.getElementById('sum-ft-drury');
        const sumDruryPct = document.getElementById('sum-drury-pct');

        const sumBaseExcl = document.getElementById('sum-base-excl');
        const sumFtExcl = document.getElementById('sum-ft-excl');
        const sumExclPct = document.getElementById('sum-excl-pct');

        const sumBaseTtr = document.getElementById('sum-base-ttr');
        const sumFtTtr = document.getElementById('sum-ft-ttr');

        const sumBaseSpeed = document.getElementById('sum-base-speed');
        const sumFtSpeed = document.getElementById('sum-ft-speed');
        const sumSpeedPct = document.getElementById('sum-speed-pct');

        const sceneList = document.getElementById('scene-selector-list');
        const comparisonView = document.getElementById('comparison-details-view');
        const vocabContainer = document.getElementById('vocab-grid-container');
        const vocabCoverageRatio = document.getElementById('vocab-coverage-ratio');

        let activeSceneId = 1;

        // Initialize Dashboard
        function init() {{
            populateSummary();
            drawCharts();
            populateSceneSelector();
            renderComparison(activeSceneId);
            populateVocabCoverage();
        }}

        function populateSummary() {{
            const summary = data.summary;
            sumBaseDrury.innerText = summary.baseline.avg_drury_count;
            sumFtDrury.innerText = summary.finetuned.avg_drury_count;
            
            const druryIncrease = summary.baseline.avg_drury_count > 0 
                ? Math.round(((summary.finetuned.avg_drury_count - summary.baseline.avg_drury_count) / summary.baseline.avg_drury_count) * 100)
                : summary.finetuned.avg_drury_count * 100;
            sumDruryPct.innerText = `+${{druryIncrease}}% Increase in Poetic Imagery`;

            sumBaseExcl.innerText = summary.baseline.avg_exclamation_count;
            sumFtExcl.innerText = summary.finetuned.avg_exclamation_count;
            
            const exclIncrease = summary.baseline.avg_exclamation_count > 0 
                ? Math.round(((summary.finetuned.avg_exclamation_count - summary.baseline.avg_exclamation_count) / summary.baseline.avg_exclamation_count) * 100)
                : summary.finetuned.avg_exclamation_count * 100;
            sumExclPct.innerText = `+${{exclIncrease}}% Exclamation Frequency`;

            sumBaseTtr.innerText = summary.baseline.avg_ttr.toFixed(3);
            sumFtTtr.innerText = summary.finetuned.avg_ttr.toFixed(3);

            sumBaseSpeed.innerText = `${{summary.baseline.avg_speed.toFixed(1)}} t/s`;
            sumFtSpeed.innerText = `${{summary.finetuned.avg_speed.toFixed(1)}} t/s`;
            
            sumSpeedPct.innerText = `${{summary.baseline.avg_generation_time.toFixed(2)}}s vs ${{summary.finetuned.avg_generation_time.toFixed(2)}}s Avg Duration`;
        }}

        function drawCharts() {{
            const baselineResults = data.comparisons;
            const len = baselineResults.length;
            
            // Draw Druryism Bar Chart
            const chartSvg = document.getElementById('metrics-chart');
            chartSvg.innerHTML = ''; // Clear
            
            const padding = 50;
            const width = 800;
            const height = 320;
            const graphHeight = height - 2 * padding;
            const graphWidth = width - 2 * padding;
            
            // Find max value for scaling
            let maxVal = 0;
            baselineResults.forEach(item => {{
                maxVal = Math.max(maxVal, item.baseline.drury_count, item.finetuned.drury_count);
            }});
            maxVal = Math.max(5, maxVal + 1); // fallback min height

            // Draw grid lines and Y axis
            for(let i=0; i<=5; i++) {{
                const yVal = padding + graphHeight - (i * graphHeight / 5);
                const valLabel = Math.round(i * maxVal / 5);
                
                // Grid line
                const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                line.setAttribute("x1", padding);
                line.setAttribute("y1", yVal);
                line.setAttribute("x2", width - padding);
                line.setAttribute("y2", yVal);
                line.setAttribute("stroke", "rgba(255,255,255,0.05)");
                line.setAttribute("stroke-width", "1");
                chartSvg.appendChild(line);
                
                // Y label
                const txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
                txt.setAttribute("x", padding - 15);
                txt.setAttribute("y", yVal + 5);
                txt.setAttribute("fill", "#64748b");
                txt.setAttribute("font-size", "11");
                txt.setAttribute("text-anchor", "end");
                txt.textContent = valLabel;
                chartSvg.appendChild(txt);
            }}

            const barWidth = graphWidth / len;
            
            baselineResults.forEach((item, index) => {{
                const xBase = padding + index * barWidth + (barWidth / 4);
                
                // Baseline bar (Grey)
                const baseH = (item.baseline.drury_count / maxVal) * graphHeight;
                const baseBar = document.createElementNS("http://www.w3.org/2000/svg", "rect");
                baseBar.setAttribute("x", xBase);
                baseBar.setAttribute("y", padding + graphHeight - baseH);
                baseBar.setAttribute("width", barWidth / 5);
                baseBar.setAttribute("height", Math.max(2, baseH));
                baseBar.setAttribute("fill", "#64748b");
                baseBar.setAttribute("rx", "3");
                chartSvg.appendChild(baseBar);

                // Fine-tuned bar (Purple/Gradient)
                const ftH = (item.finetuned.drury_count / maxVal) * graphHeight;
                const ftBar = document.createElementNS("http://www.w3.org/2000/svg", "rect");
                ftBar.setAttribute("x", xBase + (barWidth / 4));
                ftBar.setAttribute("y", padding + graphHeight - ftH);
                ftBar.setAttribute("width", barWidth / 5);
                ftBar.setAttribute("height", Math.max(2, ftH));
                ftBar.setAttribute("fill", "url(#ft-grad)");
                ftBar.setAttribute("rx", "3");
                chartSvg.appendChild(ftBar);
                
                // X label (Scene number)
                const xLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
                xLabel.setAttribute("x", xBase + (barWidth / 4));
                xLabel.setAttribute("y", height - padding + 20);
                xLabel.setAttribute("fill", "#94a3b8");
                xLabel.setAttribute("font-size", "11");
                xLabel.setAttribute("text-anchor", "middle");
                xLabel.textContent = `S${{item.id}}`;
                chartSvg.appendChild(xLabel);
            }});

            // Definitions for gradients
            const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
            const grad = document.createElementNS("http://www.w3.org/2000/svg", "linearGradient");
            grad.setAttribute("id", "ft-grad");
            grad.setAttribute("x1", "0%");
            grad.setAttribute("y1", "100%");
            grad.setAttribute("x2", "0%");
            grad.setAttribute("y2", "0%");
            
            const stop1 = document.createElementNS("http://www.w3.org/2000/svg", "stop");
            stop1.setAttribute("offset", "0%");
            stop1.setAttribute("stop-color", "#6366f1");
            const stop2 = document.createElementNS("http://www.w3.org/2000/svg", "stop");
            stop2.setAttribute("offset", "100%");
            stop2.setAttribute("stop-color", "#c084fc");
            
            grad.appendChild(stop1);
            grad.appendChild(stop2);
            defs.appendChild(grad);
            chartSvg.appendChild(defs);

            // -------------------------------------------------------------
            // Draw Speed Chart (Y axis: baseline avg vs finetuned avg)
            // -------------------------------------------------------------
            const speedChart = document.getElementById('speed-chart');
            speedChart.innerHTML = '';
            
            const sWidth = 350;
            const sHeight = 320;
            const sGraphHeight = sHeight - 2 * padding;
            const sGraphWidth = sWidth - 2 * padding;
            
            const bAvg = data.summary.baseline.avg_speed;
            const fAvg = data.summary.finetuned.avg_speed;
            const maxSpeed = Math.max(10, bAvg, fAvg) * 1.2;
            
            // Grid lines
            for(let i=0; i<=4; i++) {{
                const yVal = padding + sGraphHeight - (i * sGraphHeight / 4);
                const valLabel = Math.round(i * maxSpeed / 4);
                
                const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                line.setAttribute("x1", padding);
                line.setAttribute("y1", yVal);
                line.setAttribute("x2", sWidth - padding);
                line.setAttribute("y2", yVal);
                line.setAttribute("stroke", "rgba(255,255,255,0.05)");
                chartSvg.appendChild(line);
                
                const txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
                txt.setAttribute("x", padding - 10);
                txt.setAttribute("y", yVal + 4);
                txt.setAttribute("fill", "#64748b");
                txt.setAttribute("font-size", "10");
                txt.setAttribute("text-anchor", "end");
                txt.textContent = valLabel;
                speedChart.appendChild(txt);
            }}

            const colW = sGraphWidth / 2;
            
            // Baseline column
            const baseColH = (bAvg / maxSpeed) * sGraphHeight;
            const bCol = document.createElementNS("http://www.w3.org/2000/svg", "rect");
            bCol.setAttribute("x", padding + (colW / 4));
            bCol.setAttribute("y", padding + sGraphHeight - baseColH);
            bCol.setAttribute("width", colW / 2);
            bCol.setAttribute("height", baseColH);
            bCol.setAttribute("fill", "#475569");
            bCol.setAttribute("rx", "4");
            speedChart.appendChild(bCol);

            const bTxt = document.createElementNS("http://www.w3.org/2000/svg", "text");
            bTxt.setAttribute("x", padding + (colW / 2));
            bTxt.setAttribute("y", sHeight - padding + 20);
            bTxt.setAttribute("fill", "#94a3b8");
            bTxt.setAttribute("font-size", "11");
            bTxt.setAttribute("text-anchor", "middle");
            bTxt.textContent = "Baseline";
            speedChart.appendChild(bTxt);

            const bValTxt = document.createElementNS("http://www.w3.org/2000/svg", "text");
            bValTxt.setAttribute("x", padding + (colW / 2));
            bValTxt.setAttribute("y", padding + sGraphHeight - baseColH - 8);
            bValTxt.setAttribute("fill", "#94a3b8");
            bValTxt.setAttribute("font-size", "12");
            bValTxt.setAttribute("font-weight", "bold");
            bValTxt.setAttribute("text-anchor", "middle");
            bValTxt.textContent = `${{bAvg.toFixed(1)}} t/s`;
            speedChart.appendChild(bValTxt);

            // Fine-tuned column
            const ftColH = (fAvg / maxSpeed) * sGraphHeight;
            const fCol = document.createElementNS("http://www.w3.org/2000/svg", "rect");
            fCol.setAttribute("x", padding + colW + (colW / 4));
            fCol.setAttribute("y", padding + sGraphHeight - ftColH);
            fCol.setAttribute("width", colW / 2);
            fCol.setAttribute("height", ftColH);
            fCol.setAttribute("fill", "url(#speed-ft-grad)");
            fCol.setAttribute("rx", "4");
            speedChart.appendChild(fCol);

            const fDefs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
            const fGrad = document.createElementNS("http://www.w3.org/2000/svg", "linearGradient");
            fGrad.setAttribute("id", "speed-ft-grad");
            fGrad.setAttribute("x1", "0%");
            fGrad.setAttribute("y1", "100%");
            fGrad.setAttribute("x2", "0%");
            fGrad.setAttribute("y2", "0%");
            
            const fStop1 = document.createElementNS("http://www.w3.org/2000/svg", "stop");
            fStop1.setAttribute("offset", "0%");
            fStop1.setAttribute("stop-color", "#818cf8");
            const fStop2 = document.createElementNS("http://www.w3.org/2000/svg", "stop");
            fStop2.setAttribute("offset", "100%");
            fStop2.setAttribute("stop-color", "#c084fc");
            
            fGrad.appendChild(fStop1);
            fGrad.appendChild(fStop2);
            fDefs.appendChild(fGrad);
            speedChart.appendChild(fDefs);

            const fTxt = document.createElementNS("http://www.w3.org/2000/svg", "text");
            fTxt.setAttribute("x", padding + colW + (colW / 2));
            fTxt.setAttribute("y", sHeight - padding + 20);
            fTxt.setAttribute("fill", "#c084fc");
            fTxt.setAttribute("font-size", "11");
            fTxt.setAttribute("font-weight", "bold");
            fTxt.setAttribute("text-anchor", "middle");
            fTxt.textContent = "Fine-Tuned";
            speedChart.appendChild(fTxt);

            const fValTxt2 = document.createElementNS("http://www.w3.org/2000/svg", "text");
            fValTxt2.setAttribute("x", padding + colW + (colW / 2));
            fValTxt2.setAttribute("y", padding + sGraphHeight - ftColH - 8);
            fValTxt2.setAttribute("fill", "#c084fc");
            fValTxt2.setAttribute("font-size", "12");
            fValTxt2.setAttribute("font-weight", "bold");
            fValTxt2.setAttribute("text-anchor", "middle");
            fValTxt2.textContent = `${{fAvg.toFixed(1)}} t/s`;
            speedChart.appendChild(fValTxt2);
        }}

        function populateSceneSelector() {{
            sceneList.innerHTML = '';
            data.comparisons.forEach(item => {{
                const btn = document.createElement('button');
                btn.className = `scene-btn ${{item.id === activeSceneId ? 'active' : ''}}`;
                btn.onclick = () => selectScene(item.id);
                
                const typeClass = `type-${{item.type.toLowerCase()}}`;
                
                btn.innerHTML = `
                    <div class="scene-badge-row">
                        <span class="scene-id-lbl">Scene ${{item.id}}</span>
                        <span class="scene-type-badge ${{typeClass}}">${{item.type}}</span>
                    </div>
                    <span class="scene-snippet">${{item.scene}}</span>
                `;
                sceneList.appendChild(btn);
            }});
        }}

        function selectScene(id) {{
            activeSceneId = id;
            
            // Update active buttons
            const btns = sceneList.getElementsByClassName('scene-btn');
            Array.from(btns).forEach((btn, idx) => {{
                if (data.comparisons[idx].id === id) {{
                    btn.classList.add('active');
                }} else {{
                    btn.classList.remove('active');
                }}
            }});
            
            renderComparison(id);
        }}

        function highlightKeywords(text) {{
            if (!text) return '';
            let words = text.split(/(\\s+)/);
            return words.map(chunk => {{
                if (/^\\s+$/.test(chunk)) return chunk;
                // strip punctuation for check
                const clean = chunk.replace(/[.,!?;:"'()]/g, "").toLowerCase();
                if (trackedDruryisms.includes(clean)) {{
                    return `<span class="drury-highlight">${{chunk}}</span>`;
                }}
                return chunk;
            }}).join('');
        }}

        function renderComparison(id) {{
            const item = data.comparisons.find(c => c.id === id);
            if (!item) return;

            const baseM = item.baseline;
            const ftM = item.finetuned;

            comparisonView.innerHTML = `
                <!-- Factual Scene Input -->
                <div class="factual-input-box">
                    <div class="factual-title">
                        <span>📹 Visual scene representation (VLM Factual Output)</span>
                    </div>
                    <div class="factual-text">${{item.scene}}</div>
                </div>

                <!-- Outputs Side-by-Side -->
                <div class="outputs-container">
                    <!-- Baseline SLM -->
                    <div class="output-box">
                        <div class="output-header">
                            <span class="output-label lbl-baseline">🤖 Baseline SLM</span>
                            <span class="badge" style="font-size: 0.65rem;">Base Model</span>
                        </div>
                        <div class="text-content">${{highlightKeywords(baseM.text)}}</div>
                        
                        <table class="scene-metrics-table">
                            <thead>
                                <tr>
                                    <th>Metric</th>
                                    <th>Value</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>Generation Speed</td>
                                    <td class="m-val">${{baseM.speed.toFixed(1)}} t/s</td>
                                </tr>
                                <tr>
                                    <td>Commentary Length</td>
                                    <td class="m-val">${{baseM.word_count}} words / ${{baseM.token_count}} tokens</td>
                                </tr>
                                <tr>
                                    <td>Druryism Count</td>
                                    <td class="m-val">${{baseM.drury_count}} (${{baseM.drury_density.toFixed(1)}}%)</td>
                                </tr>
                                <tr>
                                    <td>Dramatic Markers (!)</td>
                                    <td class="m-val">${{baseM.exclamation_count}}</td>
                                </tr>
                                <tr>
                                    <td>Vocabulary TTR</td>
                                    <td class="m-val">${{baseM.ttr.toFixed(3)}}</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>

                    <!-- Fine-Tuned (adapters) -->
                    <div class="output-box finetuned-box">
                        <div class="output-header">
                            <span class="output-label lbl-finetuned">✨ Fine-Tuned SLM (Adapters)</span>
                            <span class="badge active" style="font-size: 0.65rem; background: rgba(168, 85, 247, 0.15); border-color: rgba(168, 85, 247, 0.35); color: #d8b4fe;">Active</span>
                        </div>
                        <div class="text-content poetic-font">${{highlightKeywords(ftM.text)}}</div>
                        
                        <table class="scene-metrics-table">
                            <thead>
                                <tr>
                                    <th>Metric</th>
                                    <th>Value</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>Generation Speed</td>
                                    <td class="m-val" style="color: var(--secondary);">${{ftM.speed.toFixed(1)}} t/s</td>
                                </tr>
                                <tr>
                                    <td>Commentary Length</td>
                                    <td class="m-val" style="color: var(--secondary);">${{ftM.word_count}} words / ${{ftM.token_count}} tokens</td>
                                </tr>
                                <tr>
                                    <td>Druryism Count</td>
                                    <td class="m-val" style="color: var(--secondary); font-weight: bold;">${{ftM.drury_count}} (${{ftM.drury_density.toFixed(1)}}%)</td>
                                </tr>
                                <tr>
                                    <td>Dramatic Markers (!)</td>
                                    <td class="m-val" style="color: var(--secondary);">${{ftM.exclamation_count}}</td>
                                </tr>
                                <tr>
                                    <td>Vocabulary TTR</td>
                                    <td class="m-val" style="color: var(--secondary);">${{ftM.ttr.toFixed(3)}}</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- Ground Truth / Target Reference -->
                <div class="ground-truth-box">
                    <div class="factual-title lbl-gt">
                        <span>📝 Reference Target (Drury Writing Sample)</span>
                    </div>
                    <div class="gt-text">"${{item.ground_truth}}"</div>
                </div>
            `;
        }}

        function populateVocabCoverage() {{
            const summary = data.summary;
            const hits = summary.finetuned.all_druryisms_hit;
            const total = trackedDruryisms.length;
            const hitCount = hits.length;
            const pct = Math.round((hitCount / total) * 100);

            vocabCoverageRatio.innerText = `${{hitCount}} / ${{total}} words hit (${{pct}}%)`;

            vocabContainer.innerHTML = '';
            trackedDruryisms.forEach(word => {{
                const isHit = hits.includes(word);
                const el = document.createElement('div');
                el.className = `vocab-item ${{isHit ? 'hit' : ''}}`;
                
                if (isHit) {{
                    el.innerHTML = `<span class="check-icon">✓</span> ${{word}}`;
                }} else {{
                    el.innerText = word;
                }}
                vocabContainer.appendChild(el);
            }});
        }}

        window.onload = init;
    </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Successfully generated HTML report dashboard at '{output_path}'")

# -------------------------------------------------------------------------
# Main Execution Entrypoint
# -------------------------------------------------------------------------
def main() -> None:
    logger.info("=========================================================================")
    logger.info("           STARTING PETER DRURY STYLE SLM EVALUATION SUITE")
    logger.info("=========================================================================")
    
    # 1. Evaluate Baseline Model (Without adapters)
    logger.info("--- Phase 1: Evaluating Baseline SLM ---")
    base_config = CommentaryConfig(
        base_model_id="mlx-community/Phi-3-mini-4k-instruct-4bit",
        adapter_path="",  # Path fails target check, falling back to base model
        temperature=0.85,
        top_p=0.9,
        max_tokens=200
    )
    baseline_results = run_evaluation(base_config)
    
    # 2. Evaluate Fine-Tuned Model (With adapters)
    logger.info("--- Phase 2: Evaluating Fine-Tuned SLM ---")
    ft_config = CommentaryConfig(
        base_model_id="mlx-community/Phi-3-mini-4k-instruct-4bit",
        adapter_path="./drury_adapters",  # Load adapter tensors
        temperature=0.85,
        top_p=0.9,
        max_tokens=200
    )
    finetuned_results = run_evaluation(ft_config)
    
    # 3. Compile summary statistics
    logger.info("--- Phase 3: Processing and Compiling Metrics ---")
    summary = compile_summary(baseline_results, finetuned_results)
    
    # 4. Save results to JSON file
    comparison_data = []
    for br, ftr in zip(baseline_results, finetuned_results):
        assert br["id"] == ftr["id"]
        comparison_data.append({
            "id": br["id"],
            "type": br["type"],
            "scene": br["scene"],
            "ground_truth": br["ground_truth"],
            "baseline": br["metrics"],
            "finetuned": ftr["metrics"]
        })
        
    eval_results = {
        "summary": summary,
        "comparisons": comparison_data
    }
    
    # Create docs directory if it doesn't exist
    docs_dir = os.path.join(SCRIPT_DIR, "docs")
    os.makedirs(docs_dir, exist_ok=True)

    results_json_path = os.path.join(docs_dir, "eval_results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Evaluation metrics saved to JSON file: '{results_json_path}'")
    
    # 5. Generate HTML Report
    html_report_path = os.path.join(docs_dir, "eval_report.html")
    generate_html_report(eval_results, html_report_path)
    
    logger.info("=========================================================================")
    logger.info("           EVALUATION SUITE COMPLETED SUCCESSFULLY")
    logger.info(f" Results Saved: {results_json_path}")
    logger.info(f" Visual Report: {html_report_path}")
    logger.info("=========================================================================")

if __name__ == "__main__":
    main()
