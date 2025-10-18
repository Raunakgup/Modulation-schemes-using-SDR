import tkinter as tk
from tkinter import ttk
import sys

try:
    import pluto_only_PSK
    import pluto_only_QAM
except ImportError as e:
    # If files are not found, show an error and exit.
    root = tk.Tk()
    root.withdraw() # Hide the main window
    tk.messagebox.showerror("Import Error", 
        f"Could not find the required script files.\n\n"
        f"Please make sure 'psk_pluto_demo.py' and 'qam_pluto_demo.py' "
        f"are in the same folder as this launcher.\n\nError: {e}")
    sys.exit(1)


def run_selection():
    """Runs the selected demo's main GUI function."""
    selection = combo.get()
    
    if not selection:
        tk.messagebox.showwarning("No Selection", "Please select a modulation type.")
        return

    # Destroy the launcher window before opening the demo window
    root.destroy()

    if selection == "PSK (Phase-Shift Keying)":
        print("Launching PSK Demo...")
        pluto_only_PSK.main()
    elif selection == "QAM (Quadrature Amplitude Modulation)":
        print("Launching QAM Demo...")
        pluto_only_QAM.main()

# --- Create the main launcher GUI ---
root = tk.Tk()
root.title("SDR Demo Launcher")
root.geometry("350x150") # Set a fixed size

mainframe = ttk.Frame(root, padding="20")
mainframe.pack(fill="both", expand=True)

# Label
label = ttk.Label(mainframe, text="Select which modulation demo to run:", font=("", 11))
label.pack(pady=(0, 10))

# Dropdown (Combobox)
selection_var = tk.StringVar()
combo = ttk.Combobox(mainframe, textvariable=selection_var, state="readonly", 
                     values=["PSK (Phase-Shift Keying)", 
                             "QAM (Quadrature Amplitude Modulation)"])
combo.pack(fill="x", expand=True)
combo.current(0) # Set default selection to PSK

# Run Button
run_button = ttk.Button(mainframe, text="Run Selected Demo", command=run_selection)
run_button.pack(pady=(10, 0))

# Start the GUI event loop
root.mainloop()