# server.py
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for
import queue

job_queue = queue.Queue()
worker_running = False



import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


import pluto_only_PSK as psk_mod
import pluto_only_QAM as qam_mod


BASE = Path(__file__).parent.resolve()
RESULTS_DIR = BASE / "static" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(BASE / "static"))

# ------------------ Helpers ------------------

def now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def unique_name(prefix="res"):
    return f"{prefix}_{now_str()}_{uuid.uuid4().hex[:6]}"

def parse_list_int(s):
    if s is None:
        return []
    s = str(s).strip()
    if s == "":
        return []
  
    items = []
    for part in s.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        for sub in part.split():
            if sub:
                try:
                    items.append(int(float(sub)))
                except Exception:
                
                    pass
    return items

def parse_list_float(s):
    if s is None:
        return []
    s = str(s).strip()
    if s == "":
        return []
    out = []
    for part in s.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        for sub in part.split():
            if sub:
                try:
                    out.append(float(sub))
                except Exception:
                    pass
    return out

def parse_sync_barker(s):
    if s is None:
        return None
    ss = str(s).strip()
    if ss == "" or ss.lower() in ("none", "default"):
        return None
    parts = [p.strip() for p in ss.replace(";", ",").split(",") if p.strip()]
    arr = []
    for p in parts:
        try:
            arr.append(int(p))
        except Exception:
            raise ValueError(f"Invalid sync barker element: {p}")
    import numpy as np
    return np.array(arr, dtype=int)

def save_plot_figure(fig, basename):
    outpath = RESULTS_DIR / f"{basename}.png"
    fig.savefig(str(outpath), bbox_inches="tight")
    plt.close(fig)
    return str(outpath.relative_to(BASE))

def save_current_figure(basename):
    outpath = RESULTS_DIR / f"{basename}.png"
    plt.savefig(str(outpath), bbox_inches="tight")
    plt.close()
    return str(outpath.relative_to(BASE))

def write_text_result(basename, text):
    outpath = RESULTS_DIR / f"{basename}.txt"
    with open(outpath, "w") as f:
        f.write(text)
    return str(outpath.relative_to(BASE))

# ------------------ Background job wrappers ------------------

def run_and_save_single_frame(mod_module, params, tag_prefix):
   
    try:
        data = mod_module.run_single_frame_demo(**params)
     
        try:
            fig = plt.figure(figsize=(8,6))
   
            mod_module.plot_single_frame_results(data)

            saved = []
            for i in plt.get_fignums():
                fig = plt.figure(i)
                name = unique_name(tag_prefix)
                path = RESULTS_DIR / f"{name}.png"
                fig.savefig(path, bbox_inches="tight")
                saved.append(str(path.relative_to(BASE)))
            plt.close('all')
            return {"status": "ok", "files": saved}
        except Exception as e:
            # fallback: save textual summary
            text = f"Single-frame completed but plotting failed: {e}\n"
            text += "Data keys: " + ", ".join(map(str, data.keys()))
            return {"status": "ok_no_plot", "textfile": write_text_result(unique_name(tag_prefix), text)}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def run_and_save_sps_sweep(mod_module, params, tag_prefix):
    """Call run_sps_sweep and plot via plot_sps_results."""
    try:
        data = mod_module.run_sps_sweep(**params)
        try:
            mod_module.plot_sps_results(data)
            # save all figures produced
            saved = []
            for i in plt.get_fignums():
                fig = plt.figure(i)
                name = unique_name(tag_prefix)
                path = RESULTS_DIR / f"{name}.png"
                fig.savefig(path, bbox_inches="tight")
                saved.append(str(path.relative_to(BASE)))
            plt.close('all')
            return {"status":"ok","files": saved}
        except Exception as e:
            text = f"SPS sweep completed but plotting failed: {e}"
            return {"status":"ok_no_plot", "textfile": write_text_result(unique_name(tag_prefix), text)}
    except Exception as e:
        return {"status":"error","error": str(e)}

def run_and_save_sync_len(mod_module, params, tag_prefix):
    try:
        data = mod_module.run_sync_len_sweep(**params)
        try:
            mod_module.plot_sync_len_results(data)
            saved=[]
            for i in plt.get_fignums():
                fig = plt.figure(i)
                name = unique_name(tag_prefix)
                path = RESULTS_DIR / f"{name}.png"
                fig.savefig(path, bbox_inches="tight")
                saved.append(str(path.relative_to(BASE)))
            plt.close('all')
            return {"status":"ok","files": saved}
        except Exception as e:
            return {"status":"ok_no_plot","textfile": write_text_result(unique_name(tag_prefix), f"plot err: {e}")}
    except Exception as e:
        return {"status":"error","error": str(e)}

def run_and_save_M_sweep(mod_module, params, tag_prefix):
    try:
        data = mod_module.run_M_sweep(**params)
        try:
            mod_module.plot_M_results(data)
            saved=[]
            for i in plt.get_fignums():
                fig = plt.figure(i)
                name = unique_name(tag_prefix)
                path = RESULTS_DIR / f"{name}.png"
                fig.savefig(path, bbox_inches="tight")
                saved.append(str(path.relative_to(BASE)))
            plt.close('all')
            return {"status":"ok","files": saved}
        except Exception as e:
            return {"status":"ok_no_plot","textfile": write_text_result(unique_name(tag_prefix), f"plot err: {e}")}
    except Exception as e:
        return {"status":"error","error": str(e)}

def run_and_save_N_sweep(mod_module, params, tag_prefix):
    try:
        data = mod_module.run_N_sweep(**params)
        try:
            mod_module.plot_N_results(data)
            saved=[]
            for i in plt.get_fignums():
                fig = plt.figure(i)
                name = unique_name(tag_prefix)
                path = RESULTS_DIR / f"{name}.png"
                fig.savefig(path, bbox_inches="tight")
                saved.append(str(path.relative_to(BASE)))
            plt.close('all')
            return {"status":"ok","files": saved}
        except Exception as e:
            return {"status":"ok_no_plot","textfile": write_text_result(unique_name(tag_prefix), f"plot err: {e}")}
    except Exception as e:
        return {"status":"error","error": str(e)}

def run_and_save_monte_carlo(mod_module, params, tag_prefix):
    try:
        ser, total_err, total_sym = mod_module.monte_carlo_ser(**params)
        text = f"Monte Carlo SER={ser}\nTotal errors={total_err}\nTotal symbols={total_sym}\nParams={params}"
        return {"status":"ok","textfile": write_text_result(unique_name(tag_prefix), text)}
    except Exception as e:
        return {"status":"error","error": str(e)}

def run_and_save_find_best(mod_module, params, tag_prefix):
    try:
        results = mod_module.find_best_params_for_M(**params)
        # Save textual summary
        text = f"Find-best results:\n{results}\n"
        return {"status":"ok","textfile": write_text_result(unique_name(tag_prefix), text)}
    except Exception as e:
        return {"status":"error","error": str(e)}

# ------------------ Routes ------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/run", methods=["POST"])
def run():
    form = request.form


    global_tx_gain = float(form.get("tx_gain", psk_mod.GUI_TX_GAIN))
    global_rx_gain = float(form.get("rx_gain", psk_mod.GUI_RX_GAIN))
    global_rx_gain_mode = form.get("rx_gain_mode", psk_mod.GUI_RX_GAIN_MODE)
    global_center_freq_mhz = float(form.get("center_freq_mhz", psk_mod.GUI_CENTER_FREQ/1e6))


    started_jobs = []

    def worker_loop():
        global worker_running
        while not job_queue.empty():
            target, args, kwargs = job_queue.get()
            tag = kwargs.get("tag_prefix", "task")
            try:
                res = target(*args, **kwargs)
                write_text_result(f"log_{tag}", f"Job {tag} finished with result: {res}")
            except Exception as e:
                write_text_result(f"log_{tag}", f"Job {tag} ERROR: {e}")
        worker_running = False

    def start_worker():
        global worker_running
        if not worker_running:
            worker_running = True
            t = threading.Thread(target=worker_loop, daemon=True)
            t.start()

    def start_job(target, *args, **kwargs):
        job_queue.put((target, args, kwargs))
        started_jobs.append({"description": kwargs.get("tag_prefix","task queued")})
        start_worker()

    # --------- PSK section ----------
    if form.get("psk_enable") == "on":
        mod = psk_mod
        module_label = "PSK"
        # assemble shared params
        shared = {}
        shared["pluto_ip"] = form.get("psk_pluto_ip", mod.PLUTO_IP)
        shared["seed"] = None
        # parse many params
        try:
            # Single-frame
            if form.get("psk_run_single_frame") == "on":
                params = {
                    "Nsymbols": int(form.get("psk_sf_Nsymbols", 4000)),
                    "M": int(form.get("psk_sf_M", 16)),
                    "sps": int(form.get("psk_sf_sps", 8)),
                    "fs": float(form.get("psk_sf_fs_mhz", 1.0)) * 1e6,
                    "ch_pilot_len_bits": int(form.get("psk_sf_chpilot", 128)),
                    "sync_barker13": parse_sync_barker(form.get("psk_sf_sync", "None")),
                    "seed": None,
                    "pluto_ip": form.get("psk_pluto_ip", mod.PLUTO_IP),
                }
                start_job(run_and_save_single_frame, mod, params, tag_prefix=f"psk_sf_{now_str()}")
            # SPS sweep
            if form.get("psk_run_sps_sweep") == "on":
                sps_list = parse_list_int(form.get("psk_sps_list", "2,4,8"))
                params = {
                    "fs": float(form.get("psk_sps_fs_mhz", 1.0)) * 1e6,
                    "M": int(form.get("psk_sps_M", 16)),
                    "Nsymbols": int(form.get("psk_sps_Nsymbols", 4000)),
                    "sps_list": sps_list,
                    "ch_pilot_len_bits": int(form.get("psk_sps_chpilot", 128)),
                    "sync_len_bits": int(form.get("psk_sps_sync_len", 26)),
                    "n_trials": int(form.get("psk_sps_ntrials", 10)),
                }
                start_job(run_and_save_sps_sweep, mod, params, tag_prefix=f"psk_sps_{now_str()}")
            # Sync length sweep
            if form.get("psk_run_sync_len") == "on":
                syncs = parse_list_int(form.get("psk_sync_len_list", "26,52"))
                params = {
                    "fs": float(form.get("psk_sync_fs_mhz", 1.0)) * 1e6,
                    "M": int(form.get("psk_sync_M", 16)),
                    "Nsymbols": int(form.get("psk_sync_Nsymbols", 4000)),
                    "sync_len_list": syncs,
                    "sps": int(form.get("psk_sync_sps", 8)),
                    "ch_pilot_len_bits": int(form.get("psk_sync_chpilot", 128)),
                    "n_trials": int(form.get("psk_sync_ntrials", 10)),
                }
                start_job(run_and_save_sync_len, mod, params, tag_prefix=f"psk_sync_{now_str()}")
            # M sweep
            if form.get("psk_run_M_sweep") == "on":
                Mlist = parse_list_int(form.get("psk_M_list", "2,4,8,16"))
                params = {
                    "fs": float(form.get("psk_M_fs_mhz", 1.0)) * 1e6,
                    "M_list": Mlist,
                    "Nsymbols": int(form.get("psk_M_Nsymbols", 4000)),
                    "sps": int(form.get("psk_M_sps", 8)),
                    "ch_pilot_len_bits": int(form.get("psk_M_chpilot", 128)),
                    "sync_len_bits": int(form.get("psk_M_sync_len", 26)),
                    "n_trials": int(form.get("psk_M_ntrials", 10)),
                }
                start_job(run_and_save_M_sweep, mod, params, tag_prefix=f"psk_M_{now_str()}")
            # N sweep
            if form.get("psk_run_N_sweep") == "on":
                Nlist = parse_list_int(form.get("psk_N_list", "1000,2000,4000"))
                params = {
                    "fs": float(form.get("psk_N_fs_mhz", 1.0)) * 1e6,
                    "M": int(form.get("psk_N_M", 16)),
                    "sps": int(form.get("psk_N_sps", 8)),
                    "N_list": Nlist,
                    "ch_pilot_len_bits": int(form.get("psk_N_chpilot", 128)),
                    "sync_len_bits": int(form.get("psk_N_sync_len", 26)),
                    "n_trials": int(form.get("psk_N_ntrials", 10)),
                }
                start_job(run_and_save_N_sweep, mod, params, tag_prefix=f"psk_N_{now_str()}")
            # Monte Carlo
            if form.get("psk_run_montecarlo") == "on":
                params = {
                    "n_trials": int(form.get("psk_mc_ntrials", 10)),
                    "seed": None,
                    "M": int(form.get("psk_mc_M", 16)),
                    "sps": int(form.get("psk_mc_sps", 8)),
                    "fs": float(form.get("psk_mc_fs_mhz", 1.0)) * 1e6,
                    "Nsymbols": int(form.get("psk_mc_Nsymbols", 4000)),
                    "ch_pilot_len_bits": int(form.get("psk_mc_chpilot", 128)),
                    "sync_len_bits": int(form.get("psk_mc_sync_len", 26)),
                }
                start_job(run_and_save_monte_carlo, mod, params, tag_prefix=f"psk_mc_{now_str()}")
            # Find best params
            if form.get("psk_run_find_best") == "on":
                Mlist = parse_list_int(form.get("psk_find_M_list", "2,4,8,16"))
                sps_cands = parse_list_int(form.get("psk_find_sps_candidates", "4,8"))
                N_cands = parse_list_int(form.get("psk_find_N_candidates", "1000,2000"))
                fs_cands = parse_list_float(form.get("psk_find_fs_candidates_mhz", "1.0"))
                params = {
                    "M_list": Mlist,
                    "sps_candidates": sps_cands,
                    "N_candidates": N_cands,
                    "fs_candidates": [f * 1e6 for f in fs_cands],
                    "ch_pilot_len_bits": int(form.get("psk_find_chpilot", 128)),
                    "sync_len_bits": int(form.get("psk_find_sync_len", 26)),
                    "n_trials": int(form.get("psk_find_ntrials", 10)),
                    "top_k": int(form.get("psk_find_topk", 3)),
                }
                start_job(run_and_save_find_best, mod, params, tag_prefix=f"psk_find_{now_str()}")
        except Exception as e:
            started_jobs.append({"thread_name": "psk_collect_error", "description": f"PSK param collection error: {e}"})

    # --------- QAM section (mirrored) ----------
    if form.get("qam_enable") == "on":
        mod = qam_mod
        module_label = "QAM"
        shared = {}
        shared["pluto_ip"] = form.get("psk_pluto_ip", mod.PLUTO_IP)
        shared["seed"] = None
        try:
            if form.get("qam_run_single_frame") == "on":
                params = {
                    "Nsymbols": int(form.get("qam_sf_Nsymbols", 4000)),
                    "M": int(form.get("qam_sf_M", 16)),
                    "sps": int(form.get("qam_sf_sps", 8)),
                    "fs": float(form.get("qam_sf_fs_mhz", 1.0)) * 1e6,
                    "ch_pilot_len_bits": int(form.get("qam_sf_chpilot", 128)),
                    "sync_barker13": parse_sync_barker(form.get("qam_sf_sync", "None")),
                    "seed": None,
                    "pluto_ip": form.get("qam_pluto_ip","ip:192.168.2.1"),
                }
                start_job(run_and_save_single_frame, mod, params, tag_prefix=f"qam_sf_{now_str()}")
            if form.get("qam_run_sps_sweep") == "on":
                sps_list = parse_list_int(form.get("qam_sps_list", "2,4,8"))
                params = {
                    "fs": float(form.get("qam_sps_fs_mhz", 1.0)) * 1e6,
                    "M": int(form.get("qam_sps_M", 16)),
                    "Nsymbols": int(form.get("qam_sps_Nsymbols", 4000)),
                    "sps_list": sps_list,
                    "ch_pilot_len_bits": int(form.get("qam_sps_chpilot", 128)),
                    "sync_len_bits": int(form.get("qam_sps_sync_len", 26)),
                    "n_trials": int(form.get("qam_sps_ntrials", 10)),
                }
                start_job(run_and_save_sps_sweep, mod, params, tag_prefix=f"qam_sps_{now_str()}")
            if form.get("qam_run_sync_len") == "on":
                syncs = parse_list_int(form.get("qam_sync_len_list", "26,52"))
                params = {
                    "fs": float(form.get("qam_sync_fs_mhz", 1.0)) * 1e6,
                    "M": int(form.get("qam_sync_M", 16)),
                    "Nsymbols": int(form.get("qam_sync_Nsymbols", 4000)),
                    "sync_len_list": syncs,
                    "sps": int(form.get("qam_sync_sps", 8)),
                    "ch_pilot_len_bits": int(form.get("qam_sync_chpilot", 128)),
                    "n_trials": int(form.get("qam_sync_ntrials", 10)),
                }
                start_job(run_and_save_sync_len, mod, params, tag_prefix=f"qam_sync_{now_str()}")
            if form.get("qam_run_M_sweep") == "on":
                Mlist = parse_list_int(form.get("qam_M_list", "4,16,64"))
                params = {
                    "fs": float(form.get("qam_M_fs_mhz", 1.0)) * 1e6,
                    "M_list": Mlist,
                    "Nsymbols": int(form.get("qam_M_Nsymbols", 4000)),
                    "sps": int(form.get("qam_M_sps", 8)),
                    "ch_pilot_len_bits": int(form.get("qam_M_chpilot", 128)),
                    "sync_len_bits": int(form.get("qam_M_sync_len", 26)),
                    "n_trials": int(form.get("qam_M_ntrials", 10)),
                }
                start_job(run_and_save_M_sweep, mod, params, tag_prefix=f"qam_M_{now_str()}")
            if form.get("qam_run_N_sweep") == "on":
                Nlist = parse_list_int(form.get("qam_N_list", "1000,2000,4000"))
                params = {
                    "fs": float(form.get("qam_N_fs_mhz", 1.0)) * 1e6,
                    "M": int(form.get("qam_N_M", 16)),
                    "sps": int(form.get("qam_N_sps", 8)),
                    "N_list": Nlist,
                    "ch_pilot_len_bits": int(form.get("qam_N_chpilot", 128)),
                    "sync_len_bits": int(form.get("qam_N_sync_len", 26)),
                    "n_trials": int(form.get("qam_N_ntrials", 10)),
                }
                start_job(run_and_save_N_sweep, mod, params, tag_prefix=f"qam_N_{now_str()}")
            if form.get("qam_run_montecarlo") == "on":
                params = {
                    "n_trials": int(form.get("qam_mc_ntrials", 10)),
                    "seed": None,
                    "M": int(form.get("qam_mc_M", 16)),
                    "sps": int(form.get("qam_mc_sps", 8)),
                    "fs": float(form.get("qam_mc_fs_mhz", 1.0)) * 1e6,
                    "Nsymbols": int(form.get("qam_mc_Nsymbols", 4000)),
                    "ch_pilot_len_bits": int(form.get("qam_mc_chpilot", 128)),
                    "sync_len_bits": int(form.get("qam_mc_sync_len", 26)),
                }
                start_job(run_and_save_monte_carlo, mod, params, tag_prefix=f"qam_mc_{now_str()}")
            if form.get("qam_run_find_best") == "on":
                Mlist = parse_list_int(form.get("qam_find_M_list", "4,16,64"))
                sps_cands = parse_list_int(form.get("qam_find_sps_candidates", "4,8"))
                N_cands = parse_list_int(form.get("qam_find_N_candidates", "1000,2000"))
                fs_cands = parse_list_float(form.get("qam_find_fs_candidates_mhz", "1.0"))
                params = {
                    "M_list": Mlist,
                    "sps_candidates": sps_cands,
                    "N_candidates": N_cands,
                    "fs_candidates": [f * 1e6 for f in fs_cands],
                    "ch_pilot_len_bits": int(form.get("qam_find_chpilot", 128)),
                    "sync_len_bits": int(form.get("qam_find_sync_len", 26)),
                    "n_trials": int(form.get("qam_find_ntrials", 10)),
                    "top_k": int(form.get("qam_find_topk", 3)),
                }
                start_job(run_and_save_find_best, mod, params, tag_prefix=f"qam_find_{now_str()}")
        except Exception as e:
            started_jobs.append({"thread_name": "qam_collect_error", "description": f"QAM param collection error: {e}"})
    time.sleep(1)
    files = sorted([str(p.relative_to(BASE)) for p in RESULTS_DIR.glob("*")], reverse=True)[:50]
    return render_template("result.html", jobs=started_jobs, files=files)

if __name__ == "__main__":
    # Run dev server on port 5000, accessible from LAN
    app.run(host="0.0.0.0", port=5000, debug=True)

