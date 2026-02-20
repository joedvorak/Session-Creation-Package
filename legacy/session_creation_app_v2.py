import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import sys
from io import StringIO
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import session_organizer
import threading
import time
import os
import requests
import json
import pickle 
import tempfile 

class PrintCapture:
    """Context manager to capture print statements and redirect them to a callback"""
    def __init__(self, callback, root=None):
        self.callback = callback
        self.root = root
        self.old_stdout = None
        self.string_io = None
        self.buffer = []
        
    def __enter__(self):
        self.old_stdout = sys.stdout
        self.string_io = StringIO()
        sys.stdout = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout
        # Process any remaining buffer
        self.flush()
    
    def write(self, text):
        # Write to original stdout as well (for debugging)
        self.old_stdout.write(text)
        self.old_stdout.flush()
        
        # Add to buffer
        self.buffer.append(text)
        
        # If we have complete lines, process them
        if '\n' in text:
            self.flush()
    
    def flush(self):
        if self.buffer:
            # Join buffer and split into lines
            full_text = ''.join(self.buffer)
            self.buffer = []
            
            lines = full_text.split('\n')
            
            # Keep incomplete line in buffer
            if lines and not full_text.endswith('\n'):
                self.buffer = [lines[-1]]
                lines = lines[:-1]
            
            # Send complete lines to callback
            for line in lines:
                if line.strip():  # Only send non-empty lines
                    if self.root:
                        # Schedule callback on main thread
                        self.root.after(0, lambda l=line.strip(): self.callback(l))
                    else:
                        self.callback(line.strip())

class BaseDataDialog:
    """Base class for data loading dialogs"""
    def __init__(self, parent, title, required_columns):
        self.parent = parent
        self.result = None
        self.file_path = None
        self.df = None
        self.required_columns = required_columns
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("800x700")  # Increased height for preview section
        self.dialog.resizable(True, True)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (800 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (700 // 2)
        self.dialog.geometry(f"800x700+{x}+{y}")
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File path display and browse button
        file_path_frame = ttk.Frame(file_frame)
        file_path_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_path_frame, textvariable=self.file_path_var, state='readonly')
        file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        browse_btn = ttk.Button(file_path_frame, text="Browse...", command=self.browse_file)
        browse_btn.pack(side=tk.RIGHT)
        
        # File info display
        self.file_info_var = tk.StringVar(value="No file selected")
        file_info_label = ttk.Label(file_frame, textvariable=self.file_info_var, foreground="gray")
        file_info_label.pack(anchor=tk.W)
        
        # Column mapping section
        self.columns_frame = ttk.LabelFrame(main_frame, text="Column Mapping", padding="10")
        self.columns_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Initially disabled message
        self.no_file_label = ttk.Label(self.columns_frame, text="Please select a file first", 
                                      foreground="gray", font=("TkDefaultFont", 10, "italic"))
        self.no_file_label.pack(expand=True)
        
        # Column mapping widgets (initially hidden)
        self.mapping_frame = ttk.Frame(self.columns_frame)
        self.column_vars = {}
        self.column_combos = {}
        
        # Create column selection widgets
        for i, (key, label) in enumerate(self.required_columns.items()):
            row_frame = ttk.Frame(self.mapping_frame)
            row_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(row_frame, text=f"{label}:", width=20).pack(side=tk.LEFT)
            
            self.column_vars[key] = tk.StringVar()
            combo = ttk.Combobox(row_frame, textvariable=self.column_vars[key], state='readonly')
            combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
            combo.bind('<<ComboboxSelected>>', self.on_column_selection_changed)
            self.column_combos[key] = combo
        
        # Data Preview Section
        preview_frame = ttk.LabelFrame(main_frame, text="Data Preview", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create notebook for preview tabs
        self.preview_notebook = ttk.Notebook(preview_frame)
        self.preview_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Raw data tab (always present but initially empty)
        self.raw_data_frame = ttk.Frame(self.preview_notebook)
        self.preview_notebook.add(self.raw_data_frame, text="Raw Data")
        
        # Preview tab (added after column mapping)
        self.preview_data_frame = ttk.Frame(self.preview_notebook)
        
        # Initially show message
        self.no_preview_label = ttk.Label(preview_frame, text="Load a file to see data preview", 
                                         foreground="gray", font=("TkDefaultFont", 10, "italic"))
        self.no_preview_label.pack(expand=True)
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X)
        
        ttk.Button(buttons_frame, text="Cancel", command=self.cancel).pack(side=tk.RIGHT, padx=(5, 0))
        self.load_btn = ttk.Button(buttons_frame, text="Load Data", command=self.load_data, state='disabled')
        self.load_btn.pack(side=tk.RIGHT)
        
    def browse_file(self):
        """Open file browser dialog"""
        filetypes = [
            ("All supported", "*.xlsx;*.xls;*.csv"),
            ("Excel files", "*.xlsx;*.xls"),
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            parent=self.dialog,
            title="Select data file",
            filetypes=filetypes
        )
        
        if file_path:
            self.file_path = file_path
            self.file_path_var.set(file_path)
            self.load_file_preview()
    
    def load_file_preview(self):
        """Load file and populate column dropdowns"""
        try:
            # Read file based on extension
            if self.file_path.lower().endswith('.csv'):
                self.df = pd.read_csv(self.file_path)
            elif self.file_path.lower().endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(self.file_path)
            else:
                raise ValueError("Unsupported file format")
            
            # Update file info
            info_text = f"File loaded: {len(self.df)} rows, {len(self.df.columns)} columns"
            self.file_info_var.set(info_text)
            
            # Hide "no file" message and show mapping widgets
            self.no_file_label.pack_forget()
            self.mapping_frame.pack(fill=tk.X)
            
            # Hide "no preview" message and show preview notebook
            self.no_preview_label.pack_forget()
            self.preview_notebook.pack(fill=tk.BOTH, expand=True)
            
            # Populate column dropdowns
            column_names = [''] + list(self.df.columns)  # Empty option first
            for combo in self.column_combos.values():
                combo['values'] = column_names
                combo.set('')  # Reset selection
            
            # Show raw data preview
            self.update_raw_data_preview()
            
            # Enable load button
            self.load_btn.config(state='normal')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}", parent=self.dialog)
            self.file_info_var.set("Error loading file")
    
    def update_raw_data_preview(self):
        """Update the raw data preview table"""
        # Clear existing widgets
        for widget in self.raw_data_frame.winfo_children():
            widget.destroy()
        
        if self.df is not None:
            # Create treeview for raw data
            columns = list(self.df.columns)
            tree = ttk.Treeview(self.raw_data_frame, columns=columns, show='headings', height=8)
            
            # Configure columns
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=120, minwidth=80)
            
            # Add scrollbars
            v_scrollbar = ttk.Scrollbar(self.raw_data_frame, orient=tk.VERTICAL, command=tree.yview)
            h_scrollbar = ttk.Scrollbar(self.raw_data_frame, orient=tk.HORIZONTAL, command=tree.xview)
            tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
            
            # Pack treeview and scrollbars
            tree.grid(row=0, column=0, sticky='nsew')
            v_scrollbar.grid(row=0, column=1, sticky='ns')
            h_scrollbar.grid(row=1, column=0, sticky='ew')
            
            # Configure grid weights
            self.raw_data_frame.grid_rowconfigure(0, weight=1)
            self.raw_data_frame.grid_columnconfigure(0, weight=1)
            
            # Insert data (first 50 rows)
            for idx, row in self.df.head(50).iterrows():
                values = [str(val)[:100] if pd.notna(val) else '' for val in row]  # Truncate long values
                tree.insert('', 'end', values=values)
    
    def validate_selection(self):
        """Validate that all required columns are selected"""
        for key, var in self.column_vars.items():
            if not var.get().strip():
                return False, f"Please select a column for {self.required_columns[key]}"
        
        # Check for duplicate selections
        selected_columns = [var.get() for var in self.column_vars.values()]
        if len(selected_columns) != len(set(selected_columns)):
            return False, "Please select different columns for each field"
        
        return True, ""
    
    def load_data(self):
        """Validate selections and load data"""
        valid, message = self.validate_selection()
        if not valid:
            messagebox.showwarning("Validation Error", message, parent=self.dialog)
            return
        
        try:
            # Create column mapping
            column_mapping = {var.get(): key for key, var in self.column_vars.items()}
            
            # Store result
            self.result = {
                'file_path': self.file_path,
                'dataframe': self.df,
                'column_mapping': column_mapping,
                'selected_columns': {key: var.get() for key, var in self.column_vars.items()}
            }
            
            self.dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process data:\n{str(e)}", parent=self.dialog)
    
    def cancel(self):
        """Cancel dialog"""
        self.result = None
        self.dialog.destroy()
    
    def show(self):
        """Show dialog and wait for result"""
        self.dialog.wait_window()
        return self.result

    def on_column_selection_changed(self, event=None):
        """Handle column selection changes"""
        # Check if all columns are selected
        all_selected = all(var.get().strip() for var in self.column_vars.values())
        
        if all_selected:
            # Add preview tab if not already added
            if "Mapped Preview" not in [self.preview_notebook.tab(i, "text") for i in range(self.preview_notebook.index("end"))]:
                self.preview_notebook.add(self.preview_data_frame, text="Mapped Preview")
            
            # Update mapped preview
            self.update_mapped_preview()
        else:
            # Remove preview tab if present
            try:
                for i in range(self.preview_notebook.index("end")):
                    if self.preview_notebook.tab(i, "text") == "Mapped Preview":
                        self.preview_notebook.forget(i)
                        break
            except:
                pass
    
    def update_mapped_preview(self):
        """Update the mapped data preview table"""
        # Clear existing widgets
        for widget in self.preview_data_frame.winfo_children():
            widget.destroy()
        
        if self.df is not None:
            # Get selected columns
            selected_columns = {key: var.get() for key, var in self.column_vars.items() if var.get().strip()}
            
            if len(selected_columns) == len(self.required_columns):
                # Create treeview for mapped data
                display_columns = list(self.required_columns.values())
                tree = ttk.Treeview(self.preview_data_frame, columns=display_columns, show='headings', height=8)
                
                # Configure columns
                for col in display_columns:
                    tree.heading(col, text=col)
                    tree.column(col, width=150, minwidth=100)
                
                # Add scrollbars
                v_scrollbar = ttk.Scrollbar(self.preview_data_frame, orient=tk.VERTICAL, command=tree.yview)
                h_scrollbar = ttk.Scrollbar(self.preview_data_frame, orient=tk.HORIZONTAL, command=tree.xview)
                tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
                
                # Pack treeview and scrollbars
                tree.grid(row=0, column=0, sticky='nsew')
                v_scrollbar.grid(row=0, column=1, sticky='ns')
                h_scrollbar.grid(row=1, column=0, sticky='ew')
                
                # Configure grid weights
                self.preview_data_frame.grid_rowconfigure(0, weight=1)
                self.preview_data_frame.grid_columnconfigure(0, weight=1)
                
                # Insert mapped data (first 50 rows)
                for idx, row in self.df.head(50).iterrows():
                    values = []
                    for key, display_label in self.required_columns.items():
                        if key in selected_columns:
                            source_col = selected_columns[key]
                            val = row[source_col]
                            # Truncate long values
                            display_val = str(val)[:200] if pd.notna(val) else ''
                            values.append(display_val)
                        else:
                            values.append('')
                    tree.insert('', 'end', values=values)

class NormalPresentationsDialog(BaseDataDialog):
    def __init__(self, parent):
        required_columns = {
            'title': 'Presentation Title',
            'abstract': 'Abstract', 
            'id': 'ID Number'
        }
        super().__init__(parent, "Load Normal Presentations", required_columns)

class HybridPresentationsDialog(BaseDataDialog):
    def __init__(self, parent):
        required_columns = {
            'title': 'Presentation Title',
            'abstract': 'Abstract',
            'id': 'ID Number',
            'session': 'Hybrid Session Name'
        }
        super().__init__(parent, "Load Invited Presentations in Hybrid Sessions", required_columns)

class CommitteeDialog(BaseDataDialog):
    def __init__(self, parent):
        required_columns = {
            'name': 'Committee Name',
            'description': 'Committee Description'
        }
        super().__init__(parent, "Load Committee Information", required_columns)

class SessionCreatorApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Session Creation Tool")
        self.root.geometry("1200x900")
        self.root.resizable(True, True)
        
        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Variables for tracking data states
        self.embedding_model_var = tk.StringVar(value="sentence-transformers/all-MiniLM-L6-v2")
        self.normal_presentations_loaded = tk.BooleanVar()
        self.hybrid_presentations_loaded = tk.BooleanVar()
        self.committees_loaded = tk.BooleanVar()
        self.similarity_threshold_var = tk.DoubleVar(value=0.99)
        self.max_sessions_var = tk.IntVar(value=100)
        self.min_session_size_var = tk.IntVar(value=8)
        self.llm_choice_var = tk.StringVar(value="online")
        self.api_key_var = tk.StringVar()
        self.num_committees_var = tk.IntVar(value=3)
        self.llm_model_var = tk.StringVar()
        
        # Model state
        self.embedding_model = None
        self.model_loaded = tk.BooleanVar()
        
        # Analysis state
        self.normal_embeddings = None
        self.hybrid_embeddings = None
        self.committee_embeddings = None
        self.normal_analyzed = tk.BooleanVar()
        self.hybrid_analyzed = tk.BooleanVar()
        self.committees_analyzed = tk.BooleanVar()
        
        # Session creation state
        self.sessions_created = tk.BooleanVar()
        self.df_sessions = None
        self.labels = None
        self.metadata = None
        
        # Status components (will be set during layout creation)
        self.progress_bar = None
        self.status_text = None
        
        # Ollama models cache
        self.ollama_models = []
        
        self.create_layout_option_1()
        
    def create_layout_option_1(self):
        """Tabbed interface layout - separates workflow into logical tabs"""
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Save/Load Process State Section (always visible at top)
        state_frame = ttk.LabelFrame(main_frame, text="Process State", padding="10")
        state_frame.pack(fill=tk.X, padx=5, pady=(0, 10))
        
        state_buttons_frame = ttk.Frame(state_frame)
        state_buttons_frame.pack(fill=tk.X)
        
        save_state_btn = ttk.Button(state_buttons_frame, text="Save Process State", 
                                   command=self.save_process_state)
        save_state_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        load_state_btn = ttk.Button(state_buttons_frame, text="Load Process State", 
                                   command=self.load_process_state)
        load_state_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: Data Input
        input_tab = ttk.Frame(notebook)
        notebook.add(input_tab, text="Data Input")
        self.create_input_tab(input_tab)
        
        # Tab 2: Processing
        processing_tab = ttk.Frame(notebook)
        notebook.add(processing_tab, text="Processing")
        self.create_processing_tab(processing_tab)
        
        # Tab 3: Analysis & Export
        analysis_tab = ttk.Frame(notebook)
        notebook.add(analysis_tab, text="Analysis & Export")
        self.create_analysis_tab(analysis_tab)

        # Tab 4: Data Viewer
        viewer_tab = ttk.Frame(notebook)
        notebook.add(viewer_tab, text="Data Viewer")
        self.create_data_viewer_tab(viewer_tab)
        
        # Bottom section - Progress and Status (always visible)
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.create_status_section(bottom_frame)
        
    def save_process_state(self):
        """Save the current process state to a file"""
        try:
            # Open file save dialog
            file_path = filedialog.asksaveasfilename(
                parent=self.root,
                title="Save Process State",
                defaultextension=".state",
                filetypes=[("State files", "*.state"), ("All files", "*.*")],
                initialfile="session_creation_state.state"
            )
            
            if not file_path:
                return  # User cancelled
            
            self.log_status("Saving process state...")
            self.progress_bar.config(mode='indeterminate')
            self.progress_bar.start()
            
            # Collect all state data
            state_data = {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'version': '2.0',  # State file version for compatibility checking
                
                # GUI Variables
                'gui_variables': {
                    'embedding_model_var': self.embedding_model_var.get(),
                    'similarity_threshold_var': self.similarity_threshold_var.get(),
                    'max_sessions_var': self.max_sessions_var.get(),
                    'min_session_size_var': self.min_session_size_var.get(),
                    'llm_choice_var': self.llm_choice_var.get(),
                    'api_key_var': self.api_key_var.get(),
                    'num_committees_var': self.num_committees_var.get(),
                    'llm_model_var': self.llm_model_var.get(),
                },
                
                # Boolean states
                'boolean_states': {
                    'normal_presentations_loaded': self.normal_presentations_loaded.get(),
                    'hybrid_presentations_loaded': self.hybrid_presentations_loaded.get(),
                    'committees_loaded': self.committees_loaded.get(),
                    'model_loaded': self.model_loaded.get(),
                    'normal_analyzed': self.normal_analyzed.get(),
                    'hybrid_analyzed': self.hybrid_analyzed.get(),
                    'committees_analyzed': self.committees_analyzed.get(),
                    'sessions_created': self.sessions_created.get(),
                },
                
                # Data availability flags (what data structures exist)
                'data_flags': {
                    'has_normal_presentations_data': hasattr(self, 'normal_presentations_data') and self.normal_presentations_data is not None,
                    'has_hybrid_presentations_data': hasattr(self, 'hybrid_presentations_data') and self.hybrid_presentations_data is not None,
                    'has_committee_data': hasattr(self, 'committee_data') and self.committee_data is not None,
                    'has_normal_embeddings': hasattr(self, 'normal_embeddings') and self.normal_embeddings is not None,
                    'has_hybrid_embeddings': hasattr(self, 'hybrid_embeddings') and self.hybrid_embeddings is not None,
                    'has_committee_embeddings': hasattr(self, 'committee_embeddings') and self.committee_embeddings is not None,
                    'has_df_sessions': hasattr(self, 'df_sessions') and self.df_sessions is not None,
                    'has_labels': hasattr(self, 'labels') and self.labels is not None,
                    'has_metadata': hasattr(self, 'metadata') and self.metadata is not None,
                    'has_session_committee_matches': hasattr(self, 'session_committee_matches') and self.session_committee_matches is not None,
                }
            }
            
            # Create a temporary directory for saving data files
            temp_dir = tempfile.mkdtemp(prefix="session_state_")
            
            # Save dataframes and complex data structures
            saved_files = {}
            
            # Save normal presentations data
            if state_data['data_flags']['has_normal_presentations_data']:
                normal_dir = os.path.join(temp_dir, "normal_presentations")
                os.makedirs(normal_dir, exist_ok=True)
                
                # Save dataframes
                if 'processed_dataframe' in self.normal_presentations_data:
                    processed_path = os.path.join(normal_dir, "processed_dataframe.parquet")
                    self.normal_presentations_data['processed_dataframe'].to_parquet(processed_path, index=False)
                    saved_files['normal_processed_df'] = processed_path
                
                if 'dataframe' in self.normal_presentations_data:
                    raw_path = os.path.join(normal_dir, "raw_dataframe.parquet")
                    self.normal_presentations_data['dataframe'].to_parquet(raw_path, index=False)
                    saved_files['normal_raw_df'] = raw_path
                
                # Save metadata
                normal_metadata = {k: v for k, v in self.normal_presentations_data.items() 
                                 if k not in ['processed_dataframe', 'dataframe']}
                metadata_path = os.path.join(normal_dir, "metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(normal_metadata, f, indent=2)
                saved_files['normal_metadata'] = metadata_path
            
            # Save hybrid presentations data
            if state_data['data_flags']['has_hybrid_presentations_data']:
                hybrid_dir = os.path.join(temp_dir, "hybrid_presentations")
                os.makedirs(hybrid_dir, exist_ok=True)
                
                if 'processed_dataframe' in self.hybrid_presentations_data:
                    processed_path = os.path.join(hybrid_dir, "processed_dataframe.parquet")
                    self.hybrid_presentations_data['processed_dataframe'].to_parquet(processed_path, index=False)
                    saved_files['hybrid_processed_df'] = processed_path
                
                if 'dataframe' in self.hybrid_presentations_data:
                    raw_path = os.path.join(hybrid_dir, "raw_dataframe.parquet")
                    self.hybrid_presentations_data['dataframe'].to_parquet(raw_path, index=False)
                    saved_files['hybrid_raw_df'] = raw_path
                
                if 'sessions_dataframe' in self.hybrid_presentations_data:
                    sessions_path = os.path.join(hybrid_dir, "sessions_dataframe.parquet")
                    self.hybrid_presentations_data['sessions_dataframe'].to_parquet(sessions_path, index=False)
                    saved_files['hybrid_sessions_df'] = sessions_path
                
                hybrid_metadata = {k: v for k, v in self.hybrid_presentations_data.items() 
                                 if k not in ['processed_dataframe', 'dataframe', 'sessions_dataframe']}
                metadata_path = os.path.join(hybrid_dir, "metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(hybrid_metadata, f, indent=2)
                saved_files['hybrid_metadata'] = metadata_path
            
            # Save committee data
            if state_data['data_flags']['has_committee_data']:
                committee_dir = os.path.join(temp_dir, "committee_data")
                os.makedirs(committee_dir, exist_ok=True)
                
                if 'processed_dataframe' in self.committee_data:
                    processed_path = os.path.join(committee_dir, "processed_dataframe.parquet")
                    self.committee_data['processed_dataframe'].to_parquet(processed_path, index=False)
                    saved_files['committee_processed_df'] = processed_path
                
                if 'dataframe' in self.committee_data:
                    raw_path = os.path.join(committee_dir, "raw_dataframe.parquet")
                    self.committee_data['dataframe'].to_parquet(raw_path, index=False)
                    saved_files['committee_raw_df'] = raw_path
                
                committee_metadata = {k: v for k, v in self.committee_data.items() 
                                    if k not in ['processed_dataframe', 'dataframe']}
                metadata_path = os.path.join(committee_dir, "metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(committee_metadata, f, indent=2)
                saved_files['committee_metadata'] = metadata_path
            
            # Save embeddings
            embeddings_dir = os.path.join(temp_dir, "embeddings")
            os.makedirs(embeddings_dir, exist_ok=True)
            
            if state_data['data_flags']['has_normal_embeddings']:
                normal_emb_path = os.path.join(embeddings_dir, "normal_embeddings.parquet")
                self.normal_embeddings.to_parquet(normal_emb_path, index=False)
                saved_files['normal_embeddings'] = normal_emb_path
            
            if state_data['data_flags']['has_hybrid_embeddings']:
                hybrid_emb_path = os.path.join(embeddings_dir, "hybrid_embeddings.parquet")
                self.hybrid_embeddings.to_parquet(hybrid_emb_path, index=False)
                saved_files['hybrid_embeddings'] = hybrid_emb_path
            
            if state_data['data_flags']['has_committee_embeddings']:
                committee_emb_path = os.path.join(embeddings_dir, "committee_embeddings.parquet")
                self.committee_embeddings.to_parquet(committee_emb_path, index=False)
                saved_files['committee_embeddings'] = committee_emb_path
            
            # Save session creation results
            if state_data['data_flags']['has_df_sessions']:
                sessions_path = os.path.join(temp_dir, "df_sessions.parquet")
                self.df_sessions.to_parquet(sessions_path, index=False)
                saved_files['df_sessions'] = sessions_path
            
            if state_data['data_flags']['has_labels']:
                labels_path = os.path.join(temp_dir, "labels.pkl")
                with open(labels_path, 'wb') as f:
                    pickle.dump(self.labels, f)
                saved_files['labels'] = labels_path
            
            if state_data['data_flags']['has_metadata']:
                session_metadata_path = os.path.join(temp_dir, "session_metadata.json")
                with open(session_metadata_path, 'w') as f:
                    json.dump(self.metadata, f, indent=2)
                saved_files['session_metadata'] = session_metadata_path
            
            if state_data['data_flags']['has_session_committee_matches']:
                matches_path = os.path.join(temp_dir, "session_committee_matches.parquet")
                if isinstance(self.session_committee_matches, pd.DataFrame):
                    self.session_committee_matches.to_parquet(matches_path, index=False)
                    saved_files['session_committee_matches'] = matches_path
            
            # Store the temporary directory and file list in the state
            state_data['temp_dir'] = temp_dir
            state_data['saved_files'] = saved_files
            
            # Save the main state file
            with open(file_path, 'wb') as f:
                pickle.dump(state_data, f)
            
            self.log_status(f"✓ Process state saved successfully to {file_path}")
            
            # Count items saved
            data_items = sum(1 for flag in state_data['data_flags'].values() if flag)
            messagebox.showinfo("Save Successful", 
                              f"Process state saved successfully!\n\n"
                              f"File: {os.path.basename(file_path)}\n"
                              f"Data items saved: {data_items}\n"
                              f"Timestamp: {state_data['timestamp']}")
            
        except Exception as e:
            self.log_status(f"✗ Error saving process state: {str(e)}")
            messagebox.showerror("Save Error", f"Failed to save process state:\n{str(e)}")
        finally:
            self.progress_bar.stop()
            self.progress_bar.config(mode='determinate', value=0)

    def load_process_state(self):
        """Load a previously saved process state"""
        try:
            # Open file dialog
            file_path = filedialog.askopenfilename(
                parent=self.root,
                title="Load Process State",
                filetypes=[("State files", "*.state"), ("All files", "*.*")]
            )
            
            if not file_path:
                return  # User cancelled
            
            self.log_status("Loading process state...")
            self.progress_bar.config(mode='indeterminate')
            self.progress_bar.start()
            
            # Load the main state file
            with open(file_path, 'rb') as f:
                state_data = pickle.load(f)
            
            # Check version compatibility
            state_version = state_data.get('version', '1.0')
            if state_version != '2.0':
                messagebox.showwarning("Version Mismatch", 
                                     f"This state file was created with version {state_version}. "
                                     f"Current version is 2.0. Loading may not work correctly.")
            
            # Confirm before loading (this will reset current state)
            save_timestamp = state_data.get('timestamp', 'Unknown')
            data_items = sum(1 for flag in state_data['data_flags'].values() if flag)
            
            result = messagebox.askyesno("Confirm Load", 
                                       f"This will replace your current session state.\n\n"
                                       f"State file created: {save_timestamp}\n"
                                       f"Data items: {data_items}\n\n"
                                       f"Do you want to continue?")
            if not result:
                return
            
            # Clear current state
            self.clear_current_state()
            
            # Restore GUI variables
            gui_vars = state_data.get('gui_variables', {})
            for var_name, value in gui_vars.items():
                if hasattr(self, var_name):
                    getattr(self, var_name).set(value)
            
            # Get saved files info
            temp_dir = state_data.get('temp_dir', '')
            saved_files = state_data.get('saved_files', {})
            
            # Load normal presentations data
            if state_data['data_flags']['has_normal_presentations_data']:
                self.normal_presentations_data = {}
                
                if 'normal_processed_df' in saved_files:
                    df_path = saved_files['normal_processed_df']
                    # Update path if it's relative to temp_dir
                    if not os.path.isabs(df_path) and temp_dir:
                        df_path = os.path.join(temp_dir, df_path)
                    if os.path.exists(df_path):
                        self.normal_presentations_data['processed_dataframe'] = pd.read_parquet(df_path)
                
                if 'normal_raw_df' in saved_files:
                    df_path = saved_files['normal_raw_df']
                    if not os.path.isabs(df_path) and temp_dir:
                        df_path = os.path.join(temp_dir, df_path)
                    if os.path.exists(df_path):
                        self.normal_presentations_data['dataframe'] = pd.read_parquet(df_path)
                
                if 'normal_metadata' in saved_files:
                    metadata_path = saved_files['normal_metadata']
                    if not os.path.isabs(metadata_path) and temp_dir:
                        metadata_path = os.path.join(temp_dir, metadata_path)
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            normal_metadata = json.load(f)
                        self.normal_presentations_data.update(normal_metadata)
            
            # Load hybrid presentations data
            if state_data['data_flags']['has_hybrid_presentations_data']:
                self.hybrid_presentations_data = {}
                
                if 'hybrid_processed_df' in saved_files:
                    df_path = saved_files['hybrid_processed_df']
                    if not os.path.isabs(df_path) and temp_dir:
                        df_path = os.path.join(temp_dir, df_path)
                    if os.path.exists(df_path):
                        self.hybrid_presentations_data['processed_dataframe'] = pd.read_parquet(df_path)
                
                if 'hybrid_raw_df' in saved_files:
                    df_path = saved_files['hybrid_raw_df']
                    if not os.path.isabs(df_path) and temp_dir:
                        df_path = os.path.join(temp_dir, df_path)
                    if os.path.exists(df_path):
                        self.hybrid_presentations_data['dataframe'] = pd.read_parquet(df_path)
                
                if 'hybrid_sessions_df' in saved_files:
                    df_path = saved_files['hybrid_sessions_df']
                    if not os.path.isabs(df_path) and temp_dir:
                        df_path = os.path.join(temp_dir, df_path)
                    if os.path.exists(df_path):
                        self.hybrid_presentations_data['sessions_dataframe'] = pd.read_parquet(df_path)
                
                if 'hybrid_metadata' in saved_files:
                    metadata_path = saved_files['hybrid_metadata']
                    if not os.path.isabs(metadata_path) and temp_dir:
                        metadata_path = os.path.join(temp_dir, metadata_path)
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            hybrid_metadata = json.load(f)
                        self.hybrid_presentations_data.update(hybrid_metadata)
            
            # Load committee data
            if state_data['data_flags']['has_committee_data']:
                self.committee_data = {}
                
                if 'committee_processed_df' in saved_files:
                    df_path = saved_files['committee_processed_df']
                    if not os.path.isabs(df_path) and temp_dir:
                        df_path = os.path.join(temp_dir, df_path)
                    if os.path.exists(df_path):
                        self.committee_data['processed_dataframe'] = pd.read_parquet(df_path)
                
                if 'committee_raw_df' in saved_files:
                    df_path = saved_files['committee_raw_df']
                    if not os.path.isabs(df_path) and temp_dir:
                        df_path = os.path.join(temp_dir, df_path)
                    if os.path.exists(df_path):
                        self.committee_data['dataframe'] = pd.read_parquet(df_path)
                
                if 'committee_metadata' in saved_files:
                    metadata_path = saved_files['committee_metadata']
                    if not os.path.isabs(metadata_path) and temp_dir:
                        metadata_path = os.path.join(temp_dir, metadata_path)
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            committee_metadata = json.load(f)
                        self.committee_data.update(committee_metadata)
            
            # Load embeddings
            if state_data['data_flags']['has_normal_embeddings'] and 'normal_embeddings' in saved_files:
                emb_path = saved_files['normal_embeddings']
                if not os.path.isabs(emb_path) and temp_dir:
                    emb_path = os.path.join(temp_dir, emb_path)
                if os.path.exists(emb_path):
                    self.normal_embeddings = pd.read_parquet(emb_path)
            
            if state_data['data_flags']['has_hybrid_embeddings'] and 'hybrid_embeddings' in saved_files:
                emb_path = saved_files['hybrid_embeddings']
                if not os.path.isabs(emb_path) and temp_dir:
                    emb_path = os.path.join(temp_dir, emb_path)
                if os.path.exists(emb_path):
                    self.hybrid_embeddings = pd.read_parquet(emb_path)
            
            if state_data['data_flags']['has_committee_embeddings'] and 'committee_embeddings' in saved_files:
                emb_path = saved_files['committee_embeddings']
                if not os.path.isabs(emb_path) and temp_dir:
                    emb_path = os.path.join(temp_dir, emb_path)
                if os.path.exists(emb_path):
                    self.committee_embeddings = pd.read_parquet(emb_path)
            
            # Load session results
            if state_data['data_flags']['has_df_sessions'] and 'df_sessions' in saved_files:
                sessions_path = saved_files['df_sessions']
                if not os.path.isabs(sessions_path) and temp_dir:
                    sessions_path = os.path.join(temp_dir, sessions_path)
                if os.path.exists(sessions_path):
                    self.df_sessions = pd.read_parquet(sessions_path)
            
            if state_data['data_flags']['has_labels'] and 'labels' in saved_files:
                labels_path = saved_files['labels']
                if not os.path.isabs(labels_path) and temp_dir:
                    labels_path = os.path.join(temp_dir, labels_path)
                if os.path.exists(labels_path):
                    with open(labels_path, 'rb') as f:
                        self.labels = pickle.load(f)
            
            if state_data['data_flags']['has_metadata'] and 'session_metadata' in saved_files:
                metadata_path = saved_files['session_metadata']
                if not os.path.isabs(metadata_path) and temp_dir:
                    metadata_path = os.path.join(temp_dir, metadata_path)
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        self.metadata = json.load(f)
            
            if state_data['data_flags']['has_session_committee_matches'] and 'session_committee_matches' in saved_files:
                matches_path = saved_files['session_committee_matches']
                if not os.path.isabs(matches_path) and temp_dir:
                    matches_path = os.path.join(temp_dir, matches_path)
                if os.path.exists(matches_path):
                    self.session_committee_matches = pd.read_parquet(matches_path)
            
            # Restore boolean states
            boolean_states = state_data.get('boolean_states', {})
            for var_name, value in boolean_states.items():
                if hasattr(self, var_name):
                    getattr(self, var_name).set(value)
            
            # Update UI state based on loaded data
            self.update_ui_after_state_load()
            
            # Handle embedding model loading
            saved_model = gui_vars.get('embedding_model_var')
            if saved_model and state_data['boolean_states'].get('model_loaded', False):
                try:
                    self.log_status(f"Loading embedding model: {saved_model}")
                    with PrintCapture(self.log_status, self.root):
                        self.embedding_model = SentenceTransformer(saved_model, trust_remote_code=True)
                    self.model_loaded.set(True)
                    self.log_status(f"✓ Embedding model loaded successfully")
                except Exception as e:
                    self.log_status(f"✗ Failed to load embedding model: {str(e)}")
                    self.model_loaded.set(False)
                    messagebox.showwarning("Model Loading Failed", 
                                         f"Could not load the embedding model '{saved_model}'.\n\n"
                                         f"Error: {str(e)}")
            
            # Refresh data viewer
            if hasattr(self, 'refresh_dataframe_list'):
                self.refresh_dataframe_list()
            
            self.log_status(f"✓ Process state loaded successfully from {os.path.basename(file_path)}")
            messagebox.showinfo("Load Successful", 
                              f"Process state loaded successfully!\n\n"
                              f"File: {os.path.basename(file_path)}\n"
                              f"Original timestamp: {save_timestamp}\n"
                              f"Data items restored: {data_items}")
            
        except Exception as e:
            self.log_status(f"✗ Error loading process state: {str(e)}")
            messagebox.showerror("Load Error", f"Failed to load process state:\n{str(e)}")
        finally:
            self.progress_bar.stop()
            self.progress_bar.config(mode='determinate', value=0)

    def clear_current_state(self):
        """Clear the current application state before loading a new one"""
        # Clear data structures
        self.normal_presentations_data = None
        self.hybrid_presentations_data = None
        self.committee_data = None
        self.normal_embeddings = None
        self.hybrid_embeddings = None
        self.committee_embeddings = None
        self.df_sessions = None
        self.labels = None
        self.metadata = None
        if hasattr(self, 'session_committee_matches'):
            self.session_committee_matches = None
        
        # Clear embedding model
        self.embedding_model = None
        
        # Reset all boolean variables
        self.normal_presentations_loaded.set(False)
        self.hybrid_presentations_loaded.set(False)
        self.committees_loaded.set(False)
        self.model_loaded.set(False)
        self.normal_analyzed.set(False)
        self.hybrid_analyzed.set(False)
        self.committees_analyzed.set(False)
        self.sessions_created.set(False)

    def update_ui_after_state_load(self):
        """Update UI elements after loading state"""
        # Update status indicators
        if hasattr(self, 'normal_status_indicator'):
            if self.normal_presentations_loaded.get():
                self.normal_status_indicator.config(text="Loaded", foreground="green")
                if hasattr(self, 'normal_analyze_btn') and self.embedding_model:
                    self.normal_analyze_btn.config(state='normal')
            else:
                self.normal_status_indicator.config(text="Not Loaded", foreground="red")
        
        if hasattr(self, 'hybrid_status_indicator'):
            if self.hybrid_presentations_loaded.get():
                self.hybrid_status_indicator.config(text="Loaded", foreground="green")
                if hasattr(self, 'hybrid_analyze_btn') and self.embedding_model:
                    self.hybrid_analyze_btn.config(state='normal')
            else:
                self.hybrid_status_indicator.config(text="Not Loaded", foreground="red")
        
        if hasattr(self, 'committee_status_indicator'):
            if self.committees_loaded.get():
                self.committee_status_indicator.config(text="Loaded", foreground="green")
                if hasattr(self, 'committee_analyze_btn') and self.embedding_model:
                    self.committee_analyze_btn.config(state='normal')
            else:
                self.committee_status_indicator.config(text="Not Loaded", foreground="red")
        
        # Update model status
        if hasattr(self, 'model_status_label'):
            if self.model_loaded.get():
                self.model_status_label.config(text="Loaded", foreground="green")
                if hasattr(self, 'load_model_btn'):
                    self.load_model_btn.config(text="Reload Model")
            else:
                self.model_status_label.config(text="Not Loaded", foreground="red")
                if hasattr(self, 'load_model_btn'):
                    self.load_model_btn.config(text="Load Model")
        
        # Update analysis status indicators
        if hasattr(self, 'normal_analysis_status'):
            if self.normal_analyzed.get():
                self.normal_analysis_status.config(text="Analysis: Complete", foreground="green")
                if hasattr(self, 'normal_analyze_btn'):
                    self.normal_analyze_btn.config(text="Re-analyze")
            else:
                self.normal_analysis_status.config(text="Analysis: Not Ready", foreground="orange")
        
        if hasattr(self, 'hybrid_analysis_status'):
            if self.hybrid_analyzed.get():
                self.hybrid_analysis_status.config(text="Analysis: Complete", foreground="green")
                if hasattr(self, 'hybrid_analyze_btn'):
                    self.hybrid_analyze_btn.config(text="Re-analyze")
            else:
                self.hybrid_analysis_status.config(text="Analysis: Not Ready", foreground="orange")
        
        if hasattr(self, 'committee_analysis_status'):
            if self.committees_analyzed.get():
                self.committee_analysis_status.config(text="Analysis: Complete", foreground="green")
                if hasattr(self, 'committee_analyze_btn'):
                    self.committee_analyze_btn.config(text="Re-analyze")
            else:
                self.committee_analysis_status.config(text="Analysis: Not Ready", foreground="orange")
        
        # Update session creation button
        if hasattr(self, 'create_sessions_btn'):
            ready, message = self.check_session_creation_readiness()
            if ready:
                self.create_sessions_btn.config(state='normal')
                if hasattr(self, 'session_status_label'):
                    self.session_status_label.config(text=message, foreground="blue")
        
        # Update LLM choice dependent elements
        if hasattr(self, 'on_llm_choice_changed'):
            self.on_llm_choice_changed()

    def create_input_tab(self, parent):
        """Create the data input tab"""
        # Embedding Model Selection (top of tab)
        model_frame = ttk.LabelFrame(parent, text="Embedding Model", padding="10")
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Model selection row
        model_select_frame = ttk.Frame(model_frame)
        model_select_frame.pack(fill=tk.X, pady=(0, 5))
        
        model_combo = ttk.Combobox(model_select_frame, textvariable=self.embedding_model_var, 
                                  values=["sentence-transformers/all-MiniLM-L6-v2",
                                         "sentence-transformers/all-mpnet-base-v2",
                                         "jxm/cde-small-v1"], state='normal')
        model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.load_model_btn = ttk.Button(model_select_frame, text="Load Model", 
                                        command=self.load_embedding_model)
        self.load_model_btn.pack(side=tk.RIGHT)
        
        # Model status and note
        model_status_frame = ttk.Frame(model_frame)
        model_status_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(model_status_frame, text="Model Status:").pack(side=tk.LEFT)
        self.model_status_label = ttk.Label(model_status_frame, text="Not Loaded", foreground="red")
        self.model_status_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Note about first-time download
        note_text = "Note: Loading a model for the first time can take several minutes as the model is downloaded."
        note_label = ttk.Label(model_frame, text=note_text, foreground="gray", 
                              font=("TkDefaultFont", 8), wraplength=600)
        note_label.pack(anchor=tk.W)
        
        # Container for input sections
        inputs_container = ttk.Frame(parent)
        inputs_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Three columns for the three input types
        inputs_container.columnconfigure(0, weight=1)
        inputs_container.columnconfigure(1, weight=1)
        inputs_container.columnconfigure(2, weight=1)
        
        # Normal Presentations
        normal_frame = ttk.LabelFrame(inputs_container, text="Normal Presentations", padding="10")
        normal_frame.grid(row=0, column=0, sticky='nsew', padx=5)
        self.create_input_section(normal_frame, "normal", self.normal_presentations_loaded)
        
        # Hybrid Sessions
        hybrid_frame = ttk.LabelFrame(inputs_container, text="Invited Presentations in Hybrid Sessions", padding="10")
        hybrid_frame.grid(row=0, column=1, sticky='nsew', padx=5)
        self.create_input_section(hybrid_frame, "hybrid", self.hybrid_presentations_loaded)
        
        # Committees
        committee_frame = ttk.LabelFrame(inputs_container, text="Committees", padding="10")
        committee_frame.grid(row=0, column=2, sticky='nsew', padx=5)
        self.create_input_section(committee_frame, "committees", self.committees_loaded)
        
    def create_input_section(self, parent, input_type, loaded_var):
        """Create a standardized input section"""
        # Load button
        load_btn = ttk.Button(parent, text="Read Spreadsheet", 
                             command=lambda: self.open_data_dialog(input_type))
        load_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Status indicator
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(0, 5))
        
        status_label = ttk.Label(status_frame, text="Data Status:")
        status_label.pack(side=tk.LEFT)
        
        status_indicator = ttk.Label(status_frame, text="Not Loaded", foreground="red")
        status_indicator.pack(side=tk.LEFT, padx=(5, 0))
        
        # Store references for updating status
        if input_type == "normal":
            self.normal_status_indicator = status_indicator
        elif input_type == "hybrid":
            self.hybrid_status_indicator = status_indicator
        elif input_type == "committees":
            self.committee_status_indicator = status_indicator
        
        # Analyze button
        analyze_btn = ttk.Button(parent, text="Analyze", state='disabled',
                                command=lambda: self.analyze_data(input_type))
        analyze_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Store reference to analyze button
        if input_type == "normal":
            self.normal_analyze_btn = analyze_btn
        elif input_type == "hybrid":
            self.hybrid_analyze_btn = analyze_btn
        elif input_type == "committees":
            self.committee_analyze_btn = analyze_btn
        
        # Analysis status
        analysis_status = ttk.Label(parent, text="Analysis: Not Ready", foreground="orange")
        analysis_status.pack(pady=(0, 10))
        
        # Store reference to analysis status
        if input_type == "normal":
            self.normal_analysis_status = analysis_status
        elif input_type == "hybrid":
            self.hybrid_analysis_status = analysis_status
        elif input_type == "committees":
            self.committee_analysis_status = analysis_status
        
        # Save/Load buttons
        save_load_frame = ttk.Frame(parent)
        save_load_frame.pack(fill=tk.X)
        
        save_btn = ttk.Button(save_load_frame, text="Save Analysis", 
                             command=lambda: self.save_analysis_data(input_type))
        save_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        
        load_btn = ttk.Button(save_load_frame, text="Load Analysis", 
                             command=lambda: self.load_analysis_data(input_type))
        load_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
    
    def open_data_dialog(self, data_type):
        """Open appropriate data dialog based on type"""
        try:
            if data_type == "normal":
                dialog = NormalPresentationsDialog(self.root)
                result = dialog.show()
                if result:
                    self.handle_normal_presentations_result(result)
                    
            elif data_type == "hybrid":
                dialog = HybridPresentationsDialog(self.root)
                result = dialog.show()
                if result:
                    self.handle_hybrid_presentations_result(result)
                    
            elif data_type == "committees":
                dialog = CommitteeDialog(self.root)
                result = dialog.show()
                if result:
                    self.handle_committee_result(result)
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open dialog:\n{str(e)}")
    
    def analyze_data(self, data_type):
        """Analyze data by generating embeddings"""
        # Check if model is loaded
        if not self.embedding_model:
            messagebox.showwarning("Model Not Loaded", 
                                 "Please load an embedding model first before analyzing data.")
            return
        
        try:
            if data_type == "normal":
                self.analyze_normal_presentations()
            elif data_type == "hybrid":
                self.analyze_hybrid_presentations()
            elif data_type == "committees":
                self.analyze_committees()
                
        except Exception as e:
            self.log_status(f"✗ Error during {data_type} analysis: {str(e)}")
            messagebox.showerror("Analysis Error", f"Failed to analyze {data_type} data:\n{str(e)}")

    def analyze_normal_presentations(self):
        """Analyze normal presentations by generating embeddings"""
        if not hasattr(self, 'normal_presentations_data') or not self.normal_presentations_data:
            messagebox.showwarning("No Data", "Please load normal presentations data first.")
            return
        
        try:
            self.log_status("Starting analysis of normal presentations...")
            self.progress_bar.config(mode='indeterminate')
            self.progress_bar.start()
            
            # Disable analyze button during processing
            self.normal_analyze_btn.config(state='disabled', text="Analyzing...")
            self.normal_analysis_status.config(text="Analysis: In Progress", foreground="orange")
            
            # Get the processed dataframe that was created during loading
            df_processed = self.normal_presentations_data['processed_dataframe']
            topic_column = self.normal_presentations_data['topic_column']
            
            self.root.update()
            
            # Generate embeddings using the processed dataframe
            self.log_status("Generating embeddings for normal presentations...")
            with PrintCapture(self.log_status, self.root):
                self.normal_embeddings = session_organizer.embed_documents(
                    df_processed, topic_column, self.embedding_model
                )
            
            # Update status
            self.normal_analyzed.set(True)
            self.normal_analyze_btn.config(state='normal', text="Re-analyze")
            self.normal_analysis_status.config(text="Analysis: Complete", foreground="green")
            
            # Enable create sessions button if ready
            if hasattr(self, 'create_sessions_btn'):
                ready, message = self.check_session_creation_readiness()
                if ready:
                    self.create_sessions_btn.config(state='normal')
                    self.session_status_label.config(text=message, foreground="blue")
        
            self.log_status(f"✓ Normal presentations analysis complete. Generated {self.normal_embeddings.shape[0]} embeddings.")
            
        except Exception as e:
            self.normal_analyze_btn.config(state='normal', text="Analyze")
            self.normal_analysis_status.config(text="Analysis: Error", foreground="red")
            raise e
        finally:
            self.progress_bar.stop()
            self.progress_bar.config(mode='determinate', value=0)

    def analyze_hybrid_presentations(self):
        """Analyze hybrid presentations by generating embeddings"""
        if not hasattr(self, 'hybrid_presentations_data') or not self.hybrid_presentations_data:
            messagebox.showwarning("No Data", "Please load hybrid presentations data first.")
            return
        
        try:
            self.log_status("Starting analysis of hybrid presentations...")
            self.progress_bar.config(mode='indeterminate')
            self.progress_bar.start()
            
            # Disable analyze button during processing
            self.hybrid_analyze_btn.config(state='disabled', text="Analyzing...")
            self.hybrid_analysis_status.config(text="Analysis: In Progress", foreground="orange")
            
            # Get the processed dataframe that was created during loading
            df_processed = self.hybrid_presentations_data['processed_dataframe']
            topic_column = self.hybrid_presentations_data['topic_column']
            
            self.root.update()
            
            # Generate embeddings using the processed dataframe
            self.log_status("Generating embeddings for hybrid presentations...")
            with PrintCapture(self.log_status, self.root):
                self.hybrid_embeddings = session_organizer.embed_documents(
                    df_processed, topic_column, self.embedding_model
                )
            
            # Update status
            self.hybrid_analyzed.set(True)
            self.hybrid_analyze_btn.config(state='normal', text="Re-analyze")
            self.hybrid_analysis_status.config(text="Analysis: Complete", foreground="green")
            
            self.log_status(f"✓ Hybrid presentations analysis complete. Generated {self.hybrid_embeddings.shape[0]} embeddings.")
            
        except Exception as e:
            self.hybrid_analyze_btn.config(state='normal', text="Analyze")
            self.hybrid_analysis_status.config(text="Analysis: Error", foreground="red")
            raise e
        finally:
            self.progress_bar.stop()
            self.progress_bar.config(mode='determinate', value=0)

    def analyze_committees(self):
        """Analyze committees by generating embeddings"""
        if not hasattr(self, 'committee_data') or not self.committee_data:
            messagebox.showwarning("No Data", "Please load committee data first.")
            return
        
        try:
            self.log_status("Starting analysis of committees...")
            self.progress_bar.config(mode='indeterminate')
            self.progress_bar.start()
            
            # Disable analyze button during processing
            self.committee_analyze_btn.config(state='disabled', text="Analyzing...")
            self.committee_analysis_status.config(text="Analysis: In Progress", foreground="orange")
            
            # Get the processed dataframe that was created during loading
            df_processed = self.committee_data['processed_dataframe']
            topic_column = self.committee_data['topic_column']
            
            self.root.update()
            
            # Generate embeddings using the processed dataframe
            self.log_status("Generating embeddings for committees...")
            with PrintCapture(self.log_status, self.root):
                self.committee_embeddings = session_organizer.embed_documents(
                    df_processed, topic_column, self.embedding_model
                )
            
            # Update status
            self.committees_analyzed.set(True)
            self.committee_analyze_btn.config(state='normal', text="Re-analyze")
            self.committee_analysis_status.config(text="Analysis: Complete", foreground="green")
            
            self.log_status(f"✓ Committee analysis complete. Generated {self.committee_embeddings.shape[0]} embeddings.")
            
        except Exception as e:
            self.committee_analyze_btn.config(state='normal', text="Analyze")
            self.committee_analysis_status.config(text="Analysis: Error", foreground="red")
            raise e
        finally:
            self.progress_bar.stop()
            self.progress_bar.config(mode='determinate', value=0)

    def handle_normal_presentations_result(self, result):
        """Handle normal presentations data loading result"""
        try:
            # Use session_organizer.load_presentations to properly process the data
            # This ensures proper data cleaning and column handling
            temp_file = result['file_path']
            selected_columns = result['selected_columns']
            title_col = selected_columns['title']
            abstract_col = selected_columns['abstract']
            id_col = selected_columns['id']
            
            df_processed, title_column, abstract_column, abstract_id_column, topic_column = session_organizer.load_presentations(
                temp_file,
                Title_name=title_col,
                Abstract_name=abstract_col,
                Abstract_ID_name=id_col
            )
            
            # Store the processed data
            result['processed_dataframe'] = df_processed
            result['topic_column'] = topic_column
            result['title_column'] = title_column
            result['abstract_column'] = abstract_column
            result['abstract_id_column'] = abstract_id_column
            
            self.normal_presentations_data = result
            
            # Update status
            self.normal_status_indicator.config(text="Loaded", foreground="green")
            self.normal_presentations_loaded.set(True)
            
            # Enable analyze button if model is loaded
            if self.embedding_model:
                self.normal_analyze_btn.config(state='normal')
            
            # Log success
            filename = os.path.basename(result['file_path'])
            row_count = len(df_processed)
            self.log_status(f"✓ Loaded and processed {row_count} normal presentations from {filename}")
            
        except Exception as e:
            self.log_status(f"✗ Failed to process normal presentations: {str(e)}")
            messagebox.showerror("Error", f"Failed to process normal presentations:\n{str(e)}")
    
    def handle_hybrid_presentations_result(self, result):
        """Handle hybrid presentations data loading result"""
        try:
            # Use session_organizer.load_hybrid_sessions to properly process the data
            temp_file = result['file_path']
            selected_columns = result['selected_columns']
            title_col = selected_columns['title']
            abstract_col = selected_columns['abstract']
            id_col = selected_columns['id']
            session_col = selected_columns['session']
            
            df_processed, df_sessions, hybrid_session_col, title_column, abstract_column, abstract_id_column, topic_column = session_organizer.load_hybrid_sessions(
                temp_file,
                Session_column=session_col,
                Title_column=title_col,
                Abstract_column=abstract_col,
                Abstract_ID_column=id_col
            )
            
            # Store the processed data
            result['processed_dataframe'] = df_processed
            result['sessions_dataframe'] = df_sessions
            result['topic_column'] = topic_column
            result['title_column'] = title_column
            result['abstract_column'] = abstract_column
            result['abstract_id_column'] = abstract_id_column
            result['hybrid_session_column'] = hybrid_session_col
            
            self.hybrid_presentations_data = result
            
            # Update status
            self.hybrid_status_indicator.config(text="Loaded", foreground="green")
            self.hybrid_presentations_loaded.set(True)
            
            # Enable analyze button if model is loaded
            if self.embedding_model:
                self.hybrid_analyze_btn.config(state='normal')
            
            # Log success
            filename = os.path.basename(result['file_path'])
            row_count = len(df_processed)
            self.log_status(f"✓ Loaded and processed {row_count} hybrid presentations from {filename}")
            
        except Exception as e:
            self.log_status(f"✗ Failed to process hybrid presentations: {str(e)}")
            messagebox.showerror("Error", f"Failed to process hybrid presentations:\n{str(e)}")
    
    def handle_committee_result(self, result):
        """Handle committee data loading result"""
        try:
            # Use session_organizer.load_committees to properly process the data
            temp_file = result['file_path']
            selected_columns = result['selected_columns']
            name_col = selected_columns['name']
            desc_col = selected_columns['description']
            
            df_processed, committee_name_column, description_column, combined_column = session_organizer.load_committees(
                temp_file,
                Committee_Name_column=name_col,
                Description_column=desc_col
            )
            
            # Store the processed data
            result['processed_dataframe'] = df_processed
            result['topic_column'] = combined_column
            result['committee_name_column'] = committee_name_column
            result['description_column'] = description_column
            
            self.committee_data = result
            
            # Update status
            self.committee_status_indicator.config(text="Loaded", foreground="green")
            self.committees_loaded.set(True)
            
            # Enable analyze button if model is loaded
            if self.embedding_model:
                self.committee_analyze_btn.config(state='normal')
            
            # Log success
            filename = os.path.basename(result['file_path'])
            row_count = len(df_processed)
            self.log_status(f"✓ Loaded and processed {row_count} committees from {filename}")
            
        except Exception as e:
            self.log_status(f"✗ Failed to process committee data: {str(e)}")
            messagebox.showerror("Error", f"Failed to process committee data:\n{str(e)}")

    def load_embedding_model(self):
        """Load the selected embedding model"""
        model_name = self.embedding_model_var.get()
        
        try:
            self.log_status(f"Loading embedding model: {model_name}")
            self.log_status("This may take several minutes if downloading for the first time...")
            
            # Disable the load button during loading
            self.load_model_btn.config(state='disabled', text="Loading...")
            self.model_status_label.config(text="Loading...", foreground="orange")
            
            # Update GUI to show loading state
            self.root.update()
            
            # Load the model
            with PrintCapture(self.log_status, self.root):
                self.embedding_model = SentenceTransformer(model_name, trust_remote_code=True)

            # Get model information
            if hasattr(self.embedding_model, 'model_card_data') and self.embedding_model.model_card_data:
                base_model = getattr(self.embedding_model.model_card_data, 'base_model', 'Unknown')
                self.log_status(f"✓ Model loaded successfully")
                self.log_status(f"Base model: {base_model}")
            else:
                self.log_status(f"✓ Model loaded: {model_name}")
            
            # Update status
            self.model_loaded.set(True)
            self.model_status_label.config(text="Loaded", foreground="green")
            self.load_model_btn.config(state='normal', text="Reload Model")
            
            # Enable analyze buttons for loaded data
            if hasattr(self, 'normal_presentations_data') and self.normal_presentations_data:
                self.normal_analyze_btn.config(state='normal')
            if hasattr(self, 'hybrid_presentations_data') and self.hybrid_presentations_data:
                self.hybrid_analyze_btn.config(state='normal')
            if hasattr(self, 'committee_data') and self.committee_data:
                self.committee_analyze_btn.config(state='normal')
            
        except Exception as e:
            self.log_status(f"✗ Failed to load model: {str(e)}")
            self.model_status_label.config(text="Error", foreground="red")
            self.load_model_btn.config(state='normal', text="Load Model")
            messagebox.showerror("Model Loading Error", 
                               f"Failed to load embedding model:\n{str(e)}")
    def save_analysis_data(self, data_type):
        """Save analysis data for a specific data type"""
        try:
            # Check if data exists
            if data_type == "normal":
                if not hasattr(self, 'normal_presentations_data') or not self.normal_presentations_data:
                    messagebox.showwarning("No Data", "No normal presentations data to save.")
                    return
                if not hasattr(self, 'normal_embeddings') or self.normal_embeddings is None:
                    messagebox.showwarning("No Analysis", "Normal presentations have not been analyzed yet.")
                    return
            elif data_type == "hybrid":
                if not hasattr(self, 'hybrid_presentations_data') or not self.hybrid_presentations_data:
                    messagebox.showwarning("No Data", "No hybrid presentations data to save.")
                    return
                if not hasattr(self, 'hybrid_embeddings') or self.hybrid_embeddings is None:
                    messagebox.showwarning("No Analysis", "Hybrid presentations have not been analyzed yet.")
                    return
            elif data_type == "committees":
                if not hasattr(self, 'committee_data') or not self.committee_data:
                    messagebox.showwarning("No Data", "No committee data to save.")
                    return
                if not hasattr(self, 'committee_embeddings') or self.committee_embeddings is None:
                    messagebox.showwarning("No Analysis", "Committees have not been analyzed yet.")
                    return
            
            # Open directory selection dialog
            directory = filedialog.askdirectory(
                parent=self.root,
                title=f"Select directory to save {data_type} analysis data"
            )
            
            if not directory:
                return  # User cancelled
            
            # Create subdirectory for this data type
            save_dir = os.path.join(directory, f"{data_type}_analysis")
            os.makedirs(save_dir, exist_ok=True)
            
            self.log_status(f"Saving {data_type} analysis data to {save_dir}")
            self.progress_bar.config(mode='indeterminate')
            self.progress_bar.start()
            
            # Save based on data type
            if data_type == "normal":
                # Save processed dataframe
                processed_df = self.normal_presentations_data['processed_dataframe']
                processed_path = os.path.join(save_dir, "processed_dataframe.parquet")
                processed_df.to_parquet(processed_path, index=False)
                
                # Save raw dataframe
                raw_df = self.normal_presentations_data['dataframe']
                raw_path = os.path.join(save_dir, "raw_dataframe.parquet")
                raw_df.to_parquet(raw_path, index=False)
                
                # Save embeddings
                embeddings_path = os.path.join(save_dir, "embeddings.parquet")
                self.normal_embeddings.to_parquet(embeddings_path, index=False)
                
                # Save metadata
                metadata = {
                    'data_type': data_type,
                    'topic_column': self.normal_presentations_data['topic_column'],
                    'title_column': self.normal_presentations_data['title_column'],
                    'abstract_column': self.normal_presentations_data['abstract_column'],
                    'abstract_id_column': self.normal_presentations_data['abstract_id_column'],
                    'selected_columns': self.normal_presentations_data['selected_columns'],
                    'file_path': self.normal_presentations_data['file_path'],
                    'embedding_model': self.embedding_model_var.get(),
                    'save_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
            elif data_type == "hybrid":
                # Save processed dataframe
                processed_df = self.hybrid_presentations_data['processed_dataframe']
                processed_path = os.path.join(save_dir, "processed_dataframe.parquet")
                processed_df.to_parquet(processed_path, index=False)
                
                # Save raw dataframe
                raw_df = self.hybrid_presentations_data['dataframe']
                raw_path = os.path.join(save_dir, "raw_dataframe.parquet")
                raw_df.to_parquet(raw_path, index=False)
                
                # Save sessions dataframe
                sessions_df = self.hybrid_presentations_data['sessions_dataframe']
                sessions_path = os.path.join(save_dir, "sessions_dataframe.parquet")
                sessions_df.to_parquet(sessions_path, index=False)
                
                # Save embeddings
                embeddings_path = os.path.join(save_dir, "embeddings.parquet")
                self.hybrid_embeddings.to_parquet(embeddings_path, index=False)
                
                # Save metadata
                metadata = {
                    'data_type': data_type,
                    'topic_column': self.hybrid_presentations_data['topic_column'],
                    'title_column': self.hybrid_presentations_data['title_column'],
                    'abstract_column': self.hybrid_presentations_data['abstract_column'],
                    'abstract_id_column': self.hybrid_presentations_data['abstract_id_column'],
                    'hybrid_session_column': self.hybrid_presentations_data['hybrid_session_column'],
                    'selected_columns': self.hybrid_presentations_data['selected_columns'],
                    'file_path': self.hybrid_presentations_data['file_path'],
                    'embedding_model': self.embedding_model_var.get(),
                    'save_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
            elif data_type == "committees":
                # Save processed dataframe
                processed_df = self.committee_data['processed_dataframe']
                processed_path = os.path.join(save_dir, "processed_dataframe.parquet")
                processed_df.to_parquet(processed_path, index=False)
                
                # Save raw dataframe
                raw_df = self.committee_data['dataframe']
                raw_path = os.path.join(save_dir, "raw_dataframe.parquet")
                raw_df.to_parquet(raw_path, index=False)
                
                # Save embeddings
                embeddings_path = os.path.join(save_dir, "embeddings.parquet")
                self.committee_embeddings.to_parquet(embeddings_path, index=False)
                
                # Save metadata
                metadata = {
                    'data_type': data_type,
                    'topic_column': self.committee_data['topic_column'],
                    'committee_name_column': self.committee_data['committee_name_column'],
                    'description_column': self.committee_data['description_column'],
                    'selected_columns': self.committee_data['selected_columns'],
                    'file_path': self.committee_data['file_path'],
                    'embedding_model': self.embedding_model_var.get(),
                    'save_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
            
            # Save metadata as JSON
            import json
            metadata_path = os.path.join(save_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.log_status(f"✓ {data_type.title()} analysis data saved successfully")
            messagebox.showinfo("Save Successful", 
                            f"{data_type.title()} analysis data saved to:\n{save_dir}")
            
        except Exception as e:
            self.log_status(f"✗ Error saving {data_type} analysis data: {str(e)}")
            messagebox.showerror("Save Error", f"Failed to save {data_type} analysis data:\n{str(e)}")
        finally:
            self.progress_bar.stop()
            self.progress_bar.config(mode='determinate', value=0)

    def load_analysis_data(self, data_type):
        """Load analysis data for a specific data type"""
        try:
            # Open directory selection dialog
            directory = filedialog.askdirectory(
                parent=self.root,
                title=f"Select directory containing {data_type} analysis data"
            )
            
            if not directory:
                return  # User cancelled
            
            # Check if this looks like an analysis directory
            expected_files = ["processed_dataframe.parquet", "raw_dataframe.parquet", 
                            "embeddings.parquet", "metadata.json"]
            
            missing_files = []
            for file in expected_files:
                file_path = os.path.join(directory, file)
                if not os.path.exists(file_path):
                    missing_files.append(file)
            
            if missing_files:
                messagebox.showerror("Invalid Directory", 
                                f"Selected directory is missing required files:\n" + 
                                "\n".join(missing_files))
                return
            
            self.log_status(f"Loading {data_type} analysis data from {directory}")
            self.progress_bar.config(mode='indeterminate')
            self.progress_bar.start()
            
            # Load metadata first
            import json
            metadata_path = os.path.join(directory, "metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Verify data type matches
            if metadata.get('data_type') != data_type:
                messagebox.showerror("Data Type Mismatch", 
                                f"Selected directory contains {metadata.get('data_type', 'unknown')} data, " +
                                f"but {data_type} data was expected.")
                return
            
            # Load dataframes
            processed_path = os.path.join(directory, "processed_dataframe.parquet")
            processed_df = pd.read_parquet(processed_path)
            
            raw_path = os.path.join(directory, "raw_dataframe.parquet")
            raw_df = pd.read_parquet(raw_path)
            
            embeddings_path = os.path.join(directory, "embeddings.parquet")
            embeddings_df = pd.read_parquet(embeddings_path)
            
            # Check if embedding model needs to be loaded/updated
            saved_model = metadata.get('embedding_model')
            model_needs_loading = False
            
            if saved_model:
                if not self.embedding_model:
                    # No model currently loaded
                    model_needs_loading = True
                    self.log_status(f"No embedding model currently loaded. Will load: {saved_model}")
                elif saved_model != self.embedding_model_var.get():
                    # Different model is currently loaded
                    model_needs_loading = True
                    self.log_status(f"Different model currently loaded. Switching from '{self.embedding_model_var.get()}' to '{saved_model}'")
                else:
                    self.log_status(f"Correct embedding model already loaded: {saved_model}")
            
            # Load embedding model if needed
            if model_needs_loading and saved_model:
                try:
                    self.log_status(f"Loading embedding model: {saved_model}")
                    self.embedding_model_var.set(saved_model)
                    
                    # Load the model
                    with PrintCapture(self.log_status, self.root):
                        self.embedding_model = SentenceTransformer(saved_model, trust_remote_code=True)
                    
                    # Update model status
                    self.model_loaded.set(True)
                    self.model_status_label.config(text="Loaded", foreground="green")
                    self.log_status(f"✓ Embedding model loaded successfully")
                    
                except Exception as e:
                    self.log_status(f"✗ Failed to load embedding model: {str(e)}")
                    messagebox.showwarning("Model Loading Failed", 
                                        f"Could not load the embedding model '{saved_model}' that was used for this data.\n\n"
                                        f"Error: {str(e)}\n\n"
                                        f"Please load the correct model manually before using functions that require it.")
            
            # Load type-specific data and update application state
            if data_type == "normal":
                # Reconstruct normal_presentations_data
                self.normal_presentations_data = {
                    'processed_dataframe': processed_df,
                    'dataframe': raw_df,
                    'topic_column': metadata['topic_column'],
                    'title_column': metadata['title_column'],
                    'abstract_column': metadata['abstract_column'],
                    'abstract_id_column': metadata['abstract_id_column'],
                    'selected_columns': metadata['selected_columns'],
                    'file_path': metadata['file_path']
                }
                self.normal_embeddings = embeddings_df
                
                # Update UI state
                self.normal_presentations_loaded.set(True)
                self.normal_analyzed.set(True)
                self.normal_status_indicator.config(text="Loaded", foreground="green")
                self.normal_analysis_status.config(text="Analysis: Complete", foreground="green")
                
                # Enable buttons
                if self.embedding_model:
                    self.normal_analyze_btn.config(state='normal', text="Re-analyze")
                
            elif data_type == "hybrid":
                # Load sessions dataframe
                sessions_path = os.path.join(directory, "sessions_dataframe.parquet")
                sessions_df = pd.read_parquet(sessions_path)
                
                # Reconstruct hybrid_presentations_data
                self.hybrid_presentations_data = {
                    'processed_dataframe': processed_df,
                    'dataframe': raw_df,
                    'sessions_dataframe': sessions_df,
                    'topic_column': metadata['topic_column'],
                    'title_column': metadata['title_column'],
                    'abstract_column': metadata['abstract_column'],
                    'abstract_id_column': metadata['abstract_id_column'],
                    'hybrid_session_column': metadata['hybrid_session_column'],
                    'selected_columns': metadata['selected_columns'],
                    'file_path': metadata['file_path']
                }
                self.hybrid_embeddings = embeddings_df
                
                # Update UI state
                self.hybrid_presentations_loaded.set(True)
                self.hybrid_analyzed.set(True)
                self.hybrid_status_indicator.config(text="Loaded", foreground="green")
                self.hybrid_analysis_status.config(text="Analysis: Complete", foreground="green")
                
                # Enable buttons
                if self.embedding_model:
                    self.hybrid_analyze_btn.config(state='normal', text="Re-analyze")
                
            elif data_type == "committees":
                # Reconstruct committee_data
                self.committee_data = {
                    'processed_dataframe': processed_df,
                    'dataframe': raw_df,
                    'topic_column': metadata['topic_column'],
                    'committee_name_column': metadata['committee_name_column'],
                    'description_column': metadata['description_column'],
                    'selected_columns': metadata['selected_columns'],
                    'file_path': metadata['file_path']
                }
                self.committee_embeddings = embeddings_df
                
                # Update UI state
                self.committees_loaded.set(True)
                self.committees_analyzed.set(True)
                self.committee_status_indicator.config(text="Loaded", foreground="green")
                self.committee_analysis_status.config(text="Analysis: Complete", foreground="green")
                
                # Enable buttons
                if self.embedding_model:
                    self.committee_analyze_btn.config(state='normal', text="Re-analyze")
            
            # Check session creation readiness
            if hasattr(self, 'create_sessions_btn'):
                ready, message = self.check_session_creation_readiness()
                if ready:
                    self.create_sessions_btn.config(state='normal')
                    if hasattr(self, 'session_status_label'):
                        self.session_status_label.config(text=message, foreground="blue")
            
            # Refresh data viewer
            if hasattr(self, 'refresh_dataframe_list'):
                self.refresh_dataframe_list()
            
            save_date = metadata.get('save_timestamp', 'Unknown')
            self.log_status(f"✓ {data_type.title()} analysis data loaded successfully (saved: {save_date})")
            
            # Show success message with model info
            model_info = ""
            if saved_model:
                if self.embedding_model:
                    model_info = f"\nEmbedding model: {saved_model} ✓"
                else:
                    model_info = f"\nEmbedding model: {saved_model} (failed to load)"
            
            messagebox.showinfo("Load Successful", 
                            f"{data_type.title()} analysis data loaded successfully!\n\n" +
                            f"Processed data: {len(processed_df)} rows\n" +
                            f"Raw data: {len(raw_df)} rows\n" +
                            f"Embeddings: {len(embeddings_df)} rows\n" +
                            f"Originally saved: {save_date}" +
                            model_info)
            
        except Exception as e:
            self.log_status(f"✗ Error loading {data_type} analysis data: {str(e)}")
            messagebox.showerror("Load Error", f"Failed to load {data_type} analysis data:\n{str(e)}")
        finally:
            self.progress_bar.stop()
            self.progress_bar.config(mode='determinate', value=0)
            
    def create_status_section(self, parent):
        """Create progress bar and status text area"""
        # Progress bar
        progress_frame = ttk.Frame(parent)
        progress_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(progress_frame, text="Progress:").pack(side=tk.LEFT)
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Status text area
        self.status_text = scrolledtext.ScrolledText(parent, height=8, state='disabled')
        self.status_text.pack(fill=tk.BOTH, expand=True)
        
        # Initial welcome message
        self.log_status("Welcome to Session Creation Tool")
        self.log_status("Start by selecting and loading an embedding model.")
    
    def log_status(self, message):
        """Add a status message to the log"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        # Add to the text widget
        self.status_text.config(state='normal')
        self.status_text.insert(tk.END, formatted_message + "\n")
        self.status_text.config(state='disabled')
        
        # Auto-scroll to bottom
        self.status_text.see(tk.END)
        
        # Force GUI update
        self.root.update_idletasks()

    def create_processing_tab(self, parent):
        """Create the processing tab"""
        # Remove Duplicates Section
        duplicates_frame = ttk.LabelFrame(parent, text="Check and Remove Duplicates", padding="10")
        duplicates_frame.pack(fill=tk.X, padx=5, pady=5)
        
        threshold_frame = ttk.Frame(duplicates_frame)
        threshold_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(threshold_frame, text="Similarity Threshold:").pack(side=tk.LEFT)
        threshold_scale = ttk.Scale(threshold_frame, from_=0.0, to=1.0, 
                                   variable=self.similarity_threshold_var, orient=tk.HORIZONTAL,
                                   command=self.update_threshold_display)
        threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))
        
        self.threshold_label = ttk.Label(threshold_frame, text=f"{self.similarity_threshold_var.get():.2f}")
        self.threshold_label.pack(side=tk.RIGHT)
        
        # Status section for duplicates
        duplicates_status_frame = ttk.Frame(duplicates_frame)
        duplicates_status_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(duplicates_status_frame, text="Status:").pack(side=tk.LEFT)
        self.duplicates_status_label = ttk.Label(duplicates_status_frame, text="Ready to check for duplicates", foreground="blue")
        self.duplicates_status_label.pack(side=tk.LEFT, padx=(5, 0))
        
        self.remove_duplicates_btn = ttk.Button(duplicates_frame, text="Remove Duplicates", 
                                               command=self.remove_duplicates)
        self.remove_duplicates_btn.pack()
        
        # Session Creation Section
        session_frame = ttk.LabelFrame(parent, text="Session Creation", padding="10")
        session_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Max sessions
        max_sessions_frame = ttk.Frame(session_frame)
        max_sessions_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(max_sessions_frame, text="Maximum Sessions:").pack(side=tk.LEFT)
        max_sessions_spin = ttk.Spinbox(max_sessions_frame, from_=1, to=500, 
                                       textvariable=self.max_sessions_var, width=10)
        max_sessions_spin.pack(side=tk.RIGHT)
        
        # Min session size
        min_size_frame = ttk.Frame(session_frame)
        min_size_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(min_size_frame, text="Minimum Session Size:").pack(side=tk.LEFT)
        min_size_spin = ttk.Spinbox(min_size_frame, from_=1, to=50, 
                                   textvariable=self.min_session_size_var, width=10)
        min_size_spin.pack(side=tk.RIGHT)
        
        # Status section for session creation
        session_status_frame = ttk.Frame(session_frame)
        session_status_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(session_status_frame, text="Status:").pack(side=tk.LEFT)
        self.session_status_label = ttk.Label(session_status_frame, text="Ready to create sessions", foreground="blue")
        self.session_status_label.pack(side=tk.LEFT, padx=(5, 0))
        
        self.create_sessions_btn = ttk.Button(session_frame, text="Create Sessions", 
                                            command=self.create_sessions)
        self.create_sessions_btn.pack()

        
        # Manual Session Editing Section
        manual_frame = ttk.LabelFrame(parent, text="Manually Edited Sessions", padding="10")
        manual_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Instructions text
        instructions_text = """Manual Session Assignment Instructions:

1. In Data Input: Load desired data sources and analyze them.
2. In Processing: Run Remove Duplicates and Create Sessions.
3. In Data Viewer: Click Refresh List and select "Normal Presentations (Processed)" dataframe.
4. Click "Export to CSV" to export the dataframe.
5. Open the CSV file in a spreadsheet program (Excel, Google Sheets, etc.).
6. Edit ONLY the "Session Code" column to reflect desired session assignments.
- DO NOT EDIT other parts of the file.
- Use integer values (0, 1, 2, etc.) for session codes.
- Unassigned presentations should have Session Code = -1.
7. Save the file and click "Load Edited Session Placements" below."""
        
        instructions_label = ttk.Label(manual_frame, text=instructions_text, justify=tk.LEFT, 
                                    font=("TkDefaultFont", 9), wraplength=600)
        instructions_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Status and button frame
        manual_status_frame = ttk.Frame(manual_frame)
        manual_status_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(manual_status_frame, text="Status:").pack(side=tk.LEFT)
        self.manual_status_label = ttk.Label(manual_status_frame, text="Ready to load edited sessions", foreground="blue")
        self.manual_status_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Load button
        self.load_edited_btn = ttk.Button(manual_frame, text="Load Edited Session Placements", 
                                        command=self.load_edited_session_placements)
        self.load_edited_btn.pack()

    def load_edited_session_placements(self):
        """Load manually edited session placements from CSV file"""
        # Check prerequisites
        if not hasattr(self, 'normal_presentations_data') or not self.normal_presentations_data:
            messagebox.showwarning("No Data", "Please load normal presentations data first.")
            return
        
        # Open file dialog
        file_path = filedialog.askopenfilename(
            parent=self.root,
            title="Select Edited Session Assignments CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            self.log_status("Loading edited session assignments...")
            self.progress_bar.config(mode='indeterminate')
            self.progress_bar.start()
            
            # Disable button during processing
            self.load_edited_btn.config(state='disabled', text="Loading...")
            self.manual_status_label.config(text="Processing...", foreground="orange")
            
            # Load the CSV file
            df_edited = pd.read_csv(file_path)
            
            # Validate the dataframe structure
            required_columns = ['Session Code']  # Minimum required column
            missing_columns = [col for col in required_columns if col not in df_edited.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Validate Session Code column
            session_codes = df_edited['Session Code']
            if not pd.api.types.is_numeric_dtype(session_codes):
                raise ValueError("Session Code column must contain numeric values")
            
            # Convert Session Code to integer, handling NaN values as -1 (unassigned)
            df_edited['Session Code'] = df_edited['Session Code'].fillna(-1).astype(int)
            
            # Update the stored processed dataframe
            self.normal_presentations_data['processed_dataframe'] = df_edited
            
            # Create df_sessions using the edited assignments
            self.create_sessions_from_assignments(df_edited)
            
            # Update status
            self.manual_status_label.config(text="Edited sessions loaded successfully", foreground="green")
            self.load_edited_btn.config(state='normal', text="Load Edited Session Placements")
            
            # Update session creation state
            self.sessions_created.set(True)
            
            # Count sessions and assignments
            assigned_count = len(df_edited[df_edited['Session Code'] != -1])
            unassigned_count = len(df_edited[df_edited['Session Code'] == -1])
            unique_sessions = len(df_edited[df_edited['Session Code'] != -1]['Session Code'].unique())
            
            # Log success
            self.log_status(f"✓ Loaded edited session assignments from {os.path.basename(file_path)}")
            self.log_status(f"Created {unique_sessions} sessions with {assigned_count} assigned presentations")
            self.log_status(f"Unassigned presentations: {unassigned_count}")
            
            # Refresh data viewer
            if hasattr(self, 'refresh_dataframe_list'):
                self.refresh_dataframe_list()
            
            # Show success message
            messagebox.showinfo("Success", 
                            f"Successfully loaded edited session assignments!\n\n"
                            f"Sessions created: {unique_sessions}\n"
                            f"Presentations assigned: {assigned_count}\n"
                            f"Unassigned presentations: {unassigned_count}")
            
        except Exception as e:
            error_msg = f"Failed to load edited session assignments: {str(e)}"
            self.log_status(f"✗ {error_msg}")
            self.manual_status_label.config(text="Error occurred", foreground="red")
            self.load_edited_btn.config(state='normal', text="Load Edited Session Placements")
            messagebox.showerror("Load Error", error_msg)
        finally:
            self.progress_bar.stop()
            self.progress_bar.config(mode='determinate', value=0)

    def create_sessions_from_assignments(self, df_edited):
        """Create df_sessions from manually assigned session codes"""
        try:
            # Get unique session codes (excluding -1 for unassigned)
            assigned_df = df_edited[df_edited['Session Code'] != -1]
            
            if assigned_df.empty:
                # No sessions assigned
                self.df_sessions = pd.DataFrame()
                self.labels = pd.Series(-1, index=df_edited.index, name="Session Code")
                self.metadata = {
                    'n_clusters': 0,
                    'n_assigned_items': 0,
                    'n_unassigned_items': len(df_edited),
                    'n_total_items': len(df_edited),
                    'source': 'manually_edited'
                }
                return
            print("Creating sessions from manually assigned session codes...")
            # Group by session code to create clusters
            session_groups = assigned_df.groupby('Session Code')
            final_clusters_df_indices = []
            
            for session_code, group in session_groups:
                # Get the DataFrame indices for this session
                cluster_indices = group.index.tolist()
                final_clusters_df_indices.append(cluster_indices)
            
            # Sort clusters by session code for consistency
            final_clusters_df_indices.sort(key=lambda cluster: df_edited.loc[cluster[0], 'Session Code'])
            
            # Prepare hybrid data - preserve existing hybrid sessions in their exact locations
            hybrid_cluster_presentations = []
            hybrid_session_titles = []
            
            # Check if we have existing df_sessions with hybrid data to preserve
            if (hasattr(self, 'df_sessions') and self.df_sessions is not None and 
                not self.df_sessions.empty and 
                session_organizer.COLUMNS['HYBRID_INVITED_PRESENTATIONS'] in self.df_sessions.columns and
                session_organizer.COLUMNS['FINAL_SESSION_TITLE'] in self.df_sessions.columns):
                
                # We have existing sessions with hybrid data - preserve them by session code
                existing_sessions = self.df_sessions
                
                for cluster_indices in final_clusters_df_indices:
                    # Get the session code for this cluster
                    session_code = df_edited.loc[cluster_indices[0], 'Session Code']
                    
                    # Look for an existing session with the same session code
                    matching_session = existing_sessions[existing_sessions[session_organizer.COLUMNS['CLUSTER_ID']] == session_code]
                    
                    if not matching_session.empty:
                        # Found matching session - preserve its hybrid data
                        existing_row = matching_session.iloc[0]
                        existing_hybrid = existing_row[session_organizer.COLUMNS['HYBRID_INVITED_PRESENTATIONS']]
                        existing_title = existing_row[session_organizer.COLUMNS['FINAL_SESSION_TITLE']]
                        
                        # Handle both numpy arrays and lists
                        if isinstance(existing_hybrid, (list, np.ndarray)):
                            # Convert numpy array to list if needed
                            if isinstance(existing_hybrid, np.ndarray):
                                hybrid_presentations = existing_hybrid.tolist()
                            else:
                                hybrid_presentations = existing_hybrid
                        else:
                            hybrid_presentations = []
                        
                        hybrid_cluster_presentations.append(hybrid_presentations)
                        
                        # Keep original title if it's not the default, otherwise use default
                        if existing_title != session_organizer.UNSET_SESSION_TITLE_TEXT:
                            hybrid_session_titles.append(existing_title)
                        else:
                            hybrid_session_titles.append(session_organizer.UNSET_SESSION_TITLE_TEXT)
                    else:
                        # No matching session found, use defaults
                        hybrid_cluster_presentations.append([])
                        hybrid_session_titles.append(session_organizer.UNSET_SESSION_TITLE_TEXT)
            else:
                # No existing hybrid data, use defaults for all clusters
                hybrid_cluster_presentations = [[] for _ in final_clusters_df_indices]
                hybrid_session_titles = [session_organizer.UNSET_SESSION_TITLE_TEXT for _ in final_clusters_df_indices]
            
            # Create output structures using the existing function
            df_sessions, labels, metadata = session_organizer._create_output_structures_with_df_indices(
                final_clusters_df_indices, 
                df_edited, 
                "Session Code",
                hybrid_cluster_presentations, 
                hybrid_session_titles, 
            )
            
            # Store the results
            self.df_sessions = df_sessions
            self.labels = labels
            self.metadata = metadata
            self.metadata['source'] = 'manually_edited'  # Mark as manually edited
            
            # Update the processed dataframe with session assignments (it should already be updated)
            # but ensure consistency
            df_edited['Session Code'] = labels
            self.normal_presentations_data['processed_dataframe'] = df_edited
            
            self.log_status(f"✓ Created sessions from manual assignments")
            
            # Log preservation of hybrid data
            hybrid_sessions_count = sum(1 for hybrids in hybrid_cluster_presentations if hybrids)
            if hybrid_sessions_count > 0:
                self.log_status(f"✓ Preserved {hybrid_sessions_count} sessions with hybrid presentations")
            
        except Exception as e:
            self.log_status(f"✗ Error creating sessions from assignments: {str(e)}")
            raise e
        
    def create_analysis_tab(self, parent):
        """Create the analysis and export tab"""
        # Create Titles and Keywords
        titles_frame = ttk.LabelFrame(parent, text="Create Titles and Keywords", padding="10")
        titles_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # LLM Model Type label
        llm_type_label = ttk.Label(titles_frame, text="LLM Model Type:")
        llm_type_label.pack(anchor=tk.W, pady=(0, 5))
        
        # LLM selection
        llm_frame = ttk.Frame(titles_frame)
        llm_frame.pack(fill=tk.X, pady=(0, 10))
        
        online_radio = ttk.Radiobutton(llm_frame, text="Online", variable=self.llm_choice_var, 
                                      value="online", command=self.on_llm_choice_changed)
        online_radio.pack(side=tk.LEFT, padx=(20, 0))
        
        local_radio = ttk.Radiobutton(llm_frame, text="Local (Ollama)", variable=self.llm_choice_var, 
                                     value="local", command=self.on_llm_choice_changed)
        local_radio.pack(side=tk.LEFT)
        
        # Explanatory text
        llm_note_text = "Note: Switching between Online and Local modes will check for available models, which may take a moment."
        llm_note_label = ttk.Label(titles_frame, text=llm_note_text, foreground="gray", 
                                  font=("TkDefaultFont", 8), wraplength=600)
        llm_note_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Model selection
        model_frame = ttk.Frame(titles_frame)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT)
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.llm_model_var, state='readonly')
        self.model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        
        # API Key
        api_frame = ttk.Frame(titles_frame)
        api_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.api_key_label = ttk.Label(api_frame, text="API Key:")
        self.api_key_label.pack(side=tk.LEFT)
        self.api_entry = ttk.Entry(api_frame, textvariable=self.api_key_var, show="*")
        self.api_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Status section for title generation
        title_status_frame = ttk.Frame(titles_frame)
        title_status_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(title_status_frame, text="Status:").pack(side=tk.LEFT)
        self.title_status_label = ttk.Label(title_status_frame, text="Ready to generate titles", foreground="blue")
        self.title_status_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Initialize with online mode
        self.on_llm_choice_changed()
        
        # Assign Committees
        committees_frame = ttk.LabelFrame(parent, text="Assign Committees", padding="10")
        committees_frame.pack(fill=tk.X, padx=5, pady=5)
        
        num_committees_frame = ttk.Frame(committees_frame)
        num_committees_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(num_committees_frame, text="Number of Committees:").pack(side=tk.LEFT)
        committees_spin = ttk.Spinbox(num_committees_frame, from_=1, to=10, 
                                     textvariable=self.num_committees_var, width=5)
        committees_spin.pack(side=tk.RIGHT)
        
        assign_btn = ttk.Button(committees_frame, text="Assign Committees", 
                       command=self.assign_committees)
        assign_btn.pack()
        
        self.generate_btn = ttk.Button(titles_frame, text="Generate Titles & Keywords", 
                                      command=self.generate_titles_and_keywords)
        self.generate_btn.pack()
        
    def generate_titles_and_keywords(self):
        """Generate session titles and keywords using the selected LLM model"""
        # Check if sessions have been created
        if not self.sessions_created.get() or self.df_sessions is None:
            messagebox.showwarning("No Sessions", "Please create sessions first before generating titles and keywords.")
            return
        
        # Check if we have the required data
        if not hasattr(self, 'normal_presentations_data') or not self.normal_presentations_data:
            messagebox.showwarning("No Data", "Normal presentations data is required for title generation.")
            return
        
        # Get selected model and validate
        selected_model = self.llm_model_var.get()
        if not selected_model:
            messagebox.showwarning("No Model Selected", "Please select a model for title generation.")
            return
        
        # Validate API key for online models
        if self.llm_choice_var.get() == "online":
            api_key = self.api_key_var.get().strip()
            if not api_key:
                messagebox.showwarning("API Key Required", "Please enter an API key for online model usage.")
                return
        else:
            api_key = None  # No API key needed for local models
            
            # # Set environment variable for API key
            # import os
            # if selected_model == "gemini-2.0-flash":
            #     os.environ["GEMINI_API_KEY"] = api_key
            # # Add other API key mappings as needed
        
        try:
            self.log_status("Starting title and keyword generation...")
            self.progress_bar.config(mode='indeterminate')
            self.progress_bar.start()
            
            # Disable button during processing
            self.generate_btn.config(state='disabled', text="Generating...")
            self.title_status_label.config(text="Processing...", foreground="orange")
            
            # Get the processed dataframe and topic column
            df_presentations = self.normal_presentations_data['processed_dataframe']
            topic_column = self.normal_presentations_data['topic_column']
            
            self.root.update()
            
            model_name = selected_model
            
            self.log_status(f"Using model: {model_name}")
            self.log_status(f"Generating titles for {len(self.df_sessions)} sessions...")
            
            # Call the session_organizer function to generate titles and keywords
            with PrintCapture(self.log_status, self.root):
                df_sessions_with_titles = session_organizer.generate_session_titles_and_keywords(
                    df_sessions=self.df_sessions,
                    df_presentations=df_presentations,
                    topic_column=topic_column,
                    model_name=model_name,
                    prompt_template=None,  # Use default prompt
                    api_key=api_key
                )
            
            # Update stored sessions data
            self.df_sessions = df_sessions_with_titles
            
            # Update status
            self.title_status_label.config(text="Titles generated successfully", foreground="green")
            self.generate_btn.config(state='normal', text="Generate Titles & Keywords")
            
            # Log success
            self.log_status(f"✓ Title and keyword generation complete")
            self.log_status(f"Generated titles for {len(self.df_sessions)} sessions")
            
            # Show success message
            messagebox.showinfo("Success", 
                              f"Successfully generated titles and keywords for {len(self.df_sessions)} sessions!")
            
        except Exception as e:
            self.log_status(f"✗ Error during title generation: {str(e)}")
            self.title_status_label.config(text="Error occurred", foreground="red")
            self.generate_btn.config(state='normal', text="Generate Titles & Keywords")
            messagebox.showerror("Title Generation Error", f"Failed to generate titles and keywords:\n{str(e)}")
        finally:
            self.progress_bar.stop()
            self.progress_bar.config(mode='determinate', value=0)

    def assign_committees(self):
        """Assign committees to sessions using similarity analysis"""
        # Check if sessions have been created
        if not self.sessions_created.get() or self.df_sessions is None:
            messagebox.showwarning("No Sessions", "Please create sessions first before assigning committees.")
            return
        
        # Check if we have committee data and embeddings
        if not hasattr(self, 'committee_data') or not self.committee_data:
            messagebox.showwarning("No Committee Data", "Please load committee data first.")
            return
        
        if not hasattr(self, 'committee_embeddings') or self.committee_embeddings is None:
            messagebox.showwarning("No Committee Analysis", "Please analyze committee data first to generate embeddings.")
            return
        
        # Check if we have normal presentations data and embeddings
        if not hasattr(self, 'normal_presentations_data') or not self.normal_presentations_data:
            messagebox.showwarning("No Presentations Data", "Normal presentations data is required for committee assignment.")
            return
        
        if not hasattr(self, 'normal_embeddings') or self.normal_embeddings is None:
            messagebox.showwarning("No Presentations Analysis", "Please analyze normal presentations first to generate embeddings.")
            return
        
        try:
            self.log_status("Starting committee assignment process...")
            self.progress_bar.config(mode='indeterminate')
            self.progress_bar.start()
            
            # Get number of committees to assign
            top_n = self.num_committees_var.get()
            
            # Get the required data
            df_presentations_embeddings = self.normal_embeddings.copy()
            df_committees = self.committee_data['processed_dataframe']
            committee_embeddings = self.committee_embeddings.copy()
            
            self.root.update()
            
            self.log_status(f"Finding top {top_n} committee matches for {len(self.df_sessions)} sessions...")
            
            session_committee_matches = session_organizer.find_most_similar_committees_by_presentations(
                df_sessions=self.df_sessions,
                df_presentation_embeddings=df_presentations_embeddings,
                df_committees=df_committees,
                committee_embeddings=committee_embeddings,
                top_n=top_n
            )
            # Add committee matches to the sessions dataframe
            self.log_status("Adding committee assignments to sessions...")
            df_sessions_with_committees = session_organizer.add_committee_matches_to_clusters(
                self.df_sessions, session_committee_matches
            )
            
            # Update stored sessions data
            self.df_sessions = df_sessions_with_committees
            
            # Store the committee matches for potential export
            if not hasattr(self, 'session_committee_matches'):
                self.session_committee_matches = session_committee_matches
            else:
                self.session_committee_matches = session_committee_matches
            
            # Log success
            self.log_status(f"✓ Committee assignment complete")
            self.log_status(f"Assigned committees to {len(self.df_sessions)} sessions")
            self.log_status(f"Generated {len(session_committee_matches)} committee-session matches")
            
            # Show success message
            messagebox.showinfo("Success", 
                            f"Successfully assigned committees to {len(self.df_sessions)} sessions!\n\n"
                            f"Top {top_n} committee matches found for each session.\n"
                            f"Committee assignments have been added to the sessions data.")
            
        except Exception as e:
            self.log_status(f"✗ Error during committee assignment: {str(e)}")
            messagebox.showerror("Committee Assignment Error", f"Failed to assign committees:\n{str(e)}")
        finally:
            self.progress_bar.stop()
            self.progress_bar.config(mode='determinate', value=0)
    
    def update_threshold_display(self, value):
        """Update the threshold display label when scale changes"""
        self.threshold_label.config(text=f"{float(value):.2f}")

    def remove_duplicates(self):
        """Remove duplicate presentations based on similarity threshold"""
        # Check if we have normal presentations data and embeddings
        if not hasattr(self, 'normal_presentations_data') or not self.normal_presentations_data:
            messagebox.showwarning("No Data", "Please load normal presentations data first.")
            return
        
        if not hasattr(self, 'normal_embeddings') or self.normal_embeddings is None:
            messagebox.showwarning("No Analysis", "Please analyze normal presentations first to generate embeddings.")
            return
        
        try:
            self.log_status("Starting duplicate removal process...")
            self.progress_bar.config(mode='indeterminate')
            self.progress_bar.start()
            
            # Disable button during processing
            self.remove_duplicates_btn.config(state='disabled', text="Removing...")
            self.duplicates_status_label.config(text="Processing...", foreground="orange")
            
            # Get current threshold value
            threshold = self.similarity_threshold_var.get()
            
            # Get the processed dataframe and embeddings
            df_presentations = self.normal_presentations_data['processed_dataframe']
            df_embeddings = self.normal_embeddings
            
            self.root.update()
            
            # Log initial counts
            initial_count = len(df_presentations)
            self.log_status(f"Initial presentation count: {initial_count}")
            self.log_status(f"Using similarity threshold: {threshold:.2f}")
            
            # Remove duplicates using session_organizer function
            with PrintCapture(self.log_status, self.root):
                df_presentations_clean, df_embeddings_clean = session_organizer.remove_duplicates(
                    df_presentations, df_embeddings, self.embedding_model.similarity, threshold=threshold
                )
            
            # Update stored data with cleaned versions
            self.normal_presentations_data['processed_dataframe'] = df_presentations_clean
            self.normal_embeddings = df_embeddings_clean
            
            # Log results
            final_count = len(df_presentations_clean)
            removed_count = initial_count - final_count
            
            self.log_status(f"✓ Duplicate removal complete")
            self.log_status(f"Removed {removed_count} duplicate presentations")
            self.log_status(f"Final presentation count: {final_count}")
            
            # Update status
            if removed_count > 0:
                status_text = f"Removed {removed_count} duplicates ({final_count} remaining)"
                self.duplicates_status_label.config(text=status_text, foreground="green")
            else:
                self.duplicates_status_label.config(text="No duplicates found", foreground="green")
            
            # Re-enable button
            self.remove_duplicates_btn.config(state='normal', text="Remove Duplicates")
            
        except Exception as e:
            self.log_status(f"✗ Error during duplicate removal: {str(e)}")
            self.duplicates_status_label.config(text="Error occurred", foreground="red")
            self.remove_duplicates_btn.config(state='normal', text="Remove Duplicates")
            messagebox.showerror("Duplicate Removal Error", f"Failed to remove duplicates:\n{str(e)}")
        finally:
            self.progress_bar.stop()
            self.progress_bar.config(mode='determinate', value=0)
    
    def create_sessions(self):
        """Create sessions using the session_organizer.create_sessions_w_hybrid function"""
        # Check if we have the required data
        if not hasattr(self, 'normal_presentations_data') or not self.normal_presentations_data:
            messagebox.showwarning("No Data", "Please load normal presentations data first.")
            return
        
        if not hasattr(self, 'normal_embeddings') or self.normal_embeddings is None:
            messagebox.showwarning("No Analysis", "Please analyze normal presentations first to generate embeddings.")
            return
        
        if not self.embedding_model:
            messagebox.showwarning("No Model", "Please load an embedding model first.")
            return
        
        try:
            self.log_status("Starting session creation process...")
            self.progress_bar.config(mode='indeterminate')
            self.progress_bar.start()
            
            # Disable button during processing
            self.create_sessions_btn.config(state='disabled', text="Creating Sessions...")
            self.session_status_label.config(text="Processing...", foreground="orange")
            
            # Get current parameters
            max_sessions = self.max_sessions_var.get()
            min_session_size = self.min_session_size_var.get()
            
            # Get the processed dataframe and embeddings
            df_presentations = self.normal_presentations_data['processed_dataframe']
            df_embeddings = self.normal_embeddings
            
            self.root.update()
            
            # Log parameters
            self.log_status(f"Parameters: max_sessions={max_sessions}, min_session_size={min_session_size}")
            self.log_status(f"Processing {len(df_presentations)} presentations...")
            
            # Prepare hybrid data if available
            df_hybrid_presentations = None
            hybrid_session_column = None
            df_hybrid_embeddings = None
            
            if (hasattr(self, 'hybrid_presentations_data') and self.hybrid_presentations_data and
                hasattr(self, 'hybrid_embeddings') and self.hybrid_embeddings is not None):
                
                df_hybrid_presentations = self.hybrid_presentations_data['processed_dataframe']
                hybrid_session_column = self.hybrid_presentations_data['hybrid_session_column']
                df_hybrid_embeddings = self.hybrid_embeddings
                
                self.log_status(f"Including {len(df_hybrid_presentations)} hybrid presentations...")
            
            # Create sessions using session_organizer
            with PrintCapture(self.log_status, self.root):
                df_sessions, labels, metadata = session_organizer.create_sessions_w_hybrid(
                    df_presentations=df_presentations,
                    similarity_func=self.embedding_model.similarity,
                    df_presentation_embeddings=df_embeddings,
                    df_hybrid_presentations=df_hybrid_presentations,
                    hybrid_session_column=hybrid_session_column,
                    df_hybrid_embeddings=df_hybrid_embeddings,
                    max_sessions=max_sessions,
                    min_session_size=min_session_size,
                    tree_merge_stop=0.95,  # Could make this configurable later
                    cluster_column_name="Session Code",
                )
            df_presentations['Session Code'] = labels
            
            # Analyze the resulting sessions
            # Calculate session coherence and distinctiveness and add to the DataFrame
            embeddings_only = df_embeddings.drop(columns=[session_organizer.COLUMNS['EMBEDDING_MODEL']])
            pres_similarities_matrix = self.embedding_model.similarity(embeddings_only.values, embeddings_only.values)
            # Convert to numpy if needed
            if hasattr(pres_similarities_matrix, 'cpu'):
                pres_similarities_matrix = pres_similarities_matrix.cpu().numpy()
            elif hasattr(pres_similarities_matrix, 'numpy'):
                pres_similarities_matrix = pres_similarities_matrix.numpy()

            df_presentations['presentation_session_fit'], df_sessions['session_coherence'], df_sessions['session_distinctiveness'], df_session_session_similarity = session_organizer.calculate_placement_metrics(df_presentations, df_sessions, pres_similarities_matrix, session_column_name='Session Code')

            # Store results
            self.normal_presentations_data['processed_dataframe'] = df_presentations  # Update with session assignments
            self.df_sessions = df_sessions
            self.labels = labels
            self.metadata = metadata

            # Update session creation state
            self.sessions_created.set(True)
            
            # Log results
            self.log_status(f"✓ Session creation complete")
            self.log_status(f"Created {metadata['n_clusters']} sessions")
            self.log_status(f"Assigned {metadata['n_assigned_items']} presentations")
            self.log_status(f"Unassigned presentations: {metadata['n_unassigned_items']}")
            
            # Update status
            status_text = f"Created {metadata['n_clusters']} sessions"
            self.session_status_label.config(text=status_text, foreground="green")
            
            # Re-enable button
            self.create_sessions_btn.config(state='normal', text="Create Sessions")
            
            # Show success message with details
            result_message = (
                f"Session creation completed successfully!\n\n"
                f"Sessions created: {metadata['n_clusters']}\n"
                f"Presentations assigned: {metadata['n_assigned_items']}\n"
                f"Unassigned presentations: {metadata['n_unassigned_items']}"
            )
            
            messagebox.showinfo("Success", result_message)
            
        except Exception as e:
            self.log_status(f"✗ Error during session creation: {str(e)}")
            self.session_status_label.config(text="Error occurred", foreground="red")
            self.create_sessions_btn.config(state='normal', text="Create Sessions")
            messagebox.showerror("Session Creation Error", f"Failed to create sessions:\n{str(e)}")
        finally:
            self.progress_bar.stop()
            self.progress_bar.config(mode='determinate', value=0)

    def check_session_creation_readiness(self):
        """Check if all requirements for session creation are met"""
        if not hasattr(self, 'normal_presentations_data') or not self.normal_presentations_data:
            return False, "Normal presentations data not loaded"
        
        if not hasattr(self, 'normal_embeddings') or self.normal_embeddings is None:
            return False, "Normal presentations not analyzed (embeddings missing)"
        
        if not self.embedding_model:
            return False, "Embedding model not loaded"
        
        return True, "Ready to create sessions"

    def check_ollama_availability(self):
        """Check if Ollama server is available and get available models"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                if 'models' in models_data:
                    self.ollama_models = [model['name'] for model in models_data['models']]
                    return True, f"Ollama server available with {len(self.ollama_models)} models"
                else:
                    return False, "Ollama server responded but no models found"
            else:
                return False, f"Ollama server responded with status {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, "Cannot connect to Ollama server. Make sure to run 'ollama serve' first"
        except requests.exceptions.Timeout:
            return False, "Ollama server connection timed out"
        except Exception as e:
            return False, f"Error connecting to Ollama: {str(e)}"

    def on_llm_choice_changed(self):
        """Handle LLM choice radio button changes"""
        choice = self.llm_choice_var.get()
        
        if choice == "local":
            # Check Ollama availability
            available, message = self.check_ollama_availability()
            
            if available:
                self.log_status(f"✓ {message}")
                # Update model dropdown with Ollama models
                if hasattr(self, 'model_combo'):
                    # Prepend "ollama:" to each model name for display
                    ollama_display_models = [f"ollama:{model}" for model in self.ollama_models]
                    self.model_combo['values'] = ollama_display_models
                    if ollama_display_models:
                        # Set the first formatted model name as the default
                        self.model_combo.set(ollama_display_models[0])
                    else:
                        self.model_combo.set("")    
                # Update API key field state
                if hasattr(self, 'api_entry'):
                    self.api_entry.config(state='disabled')
                    self.api_key_label.config(text="API Key (not needed for local):", foreground="gray")
                
            else:
                # Show error and revert to online
                self.log_status(f"✗ Ollama check failed: {message}")
                messagebox.showerror("Ollama Not Available", 
                                   f"Cannot use local Ollama server:\n\n{message}\n\nReverting to online mode.")
                self.llm_choice_var.set("online")
                self.on_llm_choice_changed()  # Recursively call to set online mode
                return
        
        elif choice == "online":
            # Update model dropdown with online models
            if hasattr(self, 'model_combo'):
                online_models = ["gemini-2.0-flash", "gpt-4o-mini", "claude-3-haiku"]
                self.model_combo['values'] = online_models
                self.model_combo.set("gemini-2.0-flash")  # Default online model
            
            # Enable API key field
            if hasattr(self, 'api_entry'):
                self.api_entry.config(state='normal')
                self.api_key_label.config(text="API Key:", foreground="black")
    
    def create_data_viewer_tab(self, parent):
        """Create the data viewer tab"""
        # Top frame for controls
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # DataFrame selection
        selection_frame = ttk.LabelFrame(controls_frame, text="Select DataFrame", padding="10")
        selection_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Dropdown for dataframe selection
        df_select_frame = ttk.Frame(selection_frame)
        df_select_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(df_select_frame, text="DataFrame:").pack(side=tk.LEFT)
        self.df_selection_var = tk.StringVar()
        self.df_selection_combo = ttk.Combobox(df_select_frame, textvariable=self.df_selection_var, 
                                            state='readonly', width=40)
        self.df_selection_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))
        self.df_selection_combo.bind('<<ComboboxSelected>>', self.on_dataframe_selection_changed)
        
        # Refresh and Export buttons
        buttons_frame = ttk.Frame(selection_frame)
        buttons_frame.pack(fill=tk.X)
        
        refresh_btn = ttk.Button(buttons_frame, text="Refresh List", command=self.refresh_dataframe_list)
        refresh_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.export_csv_btn = ttk.Button(buttons_frame, text="Export to CSV", 
                                        command=self.export_selected_dataframe, state='disabled')
        self.export_csv_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Info label
        self.df_info_var = tk.StringVar(value="Select a DataFrame to view")
        info_label = ttk.Label(selection_frame, textvariable=self.df_info_var, foreground="gray")
        info_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Data display frame
        display_frame = ttk.LabelFrame(parent, text="Data Preview", padding="10")
        display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create treeview for data display
        self.data_tree = ttk.Treeview(display_frame, show='headings', height=15)
        
        # Scrollbars for the treeview
        v_scrollbar = ttk.Scrollbar(display_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        h_scrollbar = ttk.Scrollbar(display_frame, orient=tk.HORIZONTAL, command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack treeview and scrollbars
        self.data_tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        # Configure grid weights
        display_frame.grid_rowconfigure(0, weight=1)
        display_frame.grid_columnconfigure(0, weight=1)
        
        # Initially populate the dataframe list
        self.refresh_dataframe_list()
    
    def refresh_dataframe_list(self):
        """Refresh the list of available dataframes"""
        dataframes = []
        
        # Check for normal presentations data
        if hasattr(self, 'normal_presentations_data') and self.normal_presentations_data:
            if 'processed_dataframe' in self.normal_presentations_data:
                dataframes.append(("Normal Presentations (Processed)", "normal_processed"))
            if 'dataframe' in self.normal_presentations_data:
                dataframes.append(("Normal Presentations (Raw)", "normal_raw"))
        
        # Check for hybrid presentations data
        if hasattr(self, 'hybrid_presentations_data') and self.hybrid_presentations_data:
            if 'processed_dataframe' in self.hybrid_presentations_data:
                dataframes.append(("Hybrid Presentations (Processed)", "hybrid_processed"))
            if 'dataframe' in self.hybrid_presentations_data:
                dataframes.append(("Hybrid Presentations (Raw)", "hybrid_raw"))
            if 'sessions_dataframe' in self.hybrid_presentations_data:
                dataframes.append(("Hybrid Sessions", "hybrid_sessions"))
        
        # Check for committee data
        if hasattr(self, 'committee_data') and self.committee_data:
            if 'processed_dataframe' in self.committee_data:
                dataframes.append(("Committees (Processed)", "committee_processed"))
            if 'dataframe' in self.committee_data:
                dataframes.append(("Committees (Raw)", "committee_raw"))
        
        # Check for session data
        if hasattr(self, 'df_sessions') and self.df_sessions is not None:
            dataframes.append(("Created Sessions", "sessions"))

        # Check for committee matches data
        if hasattr(self, 'session_committee_matches') and self.session_committee_matches is not None:
            dataframes.append(("Session-Committee Matches", "committee_matches"))
        # Update the combobox
        if hasattr(self, 'df_selection_combo'):
            display_names = [name for name, key in dataframes]
            self.df_selection_combo['values'] = display_names
            
            # Store the mapping for later use
            self.dataframe_mapping = {name: key for name, key in dataframes}
            
            # Clear current selection if the previously selected item is no longer available
            current_selection = self.df_selection_var.get()
            if current_selection not in display_names:
                self.df_selection_var.set("")
                self.clear_data_display()
            
            # Update info
            if dataframes:
                self.df_info_var.set(f"{len(dataframes)} DataFrame(s) available")
            else:
                self.df_info_var.set("No DataFrames available - load some data first")

    def on_dataframe_selection_changed(self, event=None):
        """Handle dataframe selection change"""
        selected_name = self.df_selection_var.get()
        if not selected_name or not hasattr(self, 'dataframe_mapping'):
            return
        
        if selected_name not in self.dataframe_mapping:
            return
        
        dataframe_key = self.dataframe_mapping[selected_name]
        df = self.get_dataframe_by_key(dataframe_key)
        
        if df is not None:
            self.display_dataframe(df, selected_name)
            self.export_csv_btn.config(state='normal')
        else:
            self.clear_data_display()
            self.export_csv_btn.config(state='disabled')

    def get_dataframe_by_key(self, key):
        """Get dataframe by its key identifier"""
        try:
            if key == "normal_processed":
                return self.normal_presentations_data['processed_dataframe']
            elif key == "normal_raw":
                return self.normal_presentations_data['dataframe']
            elif key == "hybrid_processed":
                return self.hybrid_presentations_data['processed_dataframe']
            elif key == "hybrid_raw":
                return self.hybrid_presentations_data['dataframe']
            elif key == "hybrid_sessions":
                return self.hybrid_presentations_data['sessions_dataframe']
            elif key == "committee_processed":
                return self.committee_data['processed_dataframe']
            elif key == "committee_raw":
                return self.committee_data['dataframe']
            elif key == "sessions":
                return self.df_sessions
            elif key == "committee_matches":
                return self.session_committee_matches if hasattr(self, 'session_committee_matches') else None
            else:
                return None
        except (KeyError, AttributeError):
            return None

    def display_dataframe(self, df, name):
        """Display the selected dataframe in the treeview"""
        # Clear existing data
        self.clear_data_display()
        
        if df is None or df.empty:
            self.df_info_var.set(f"{name}: Empty DataFrame")
            return
        
        # Update info
        self.df_info_var.set(f"{name}: {len(df)} rows × {len(df.columns)} columns")
        
        # Set up columns
        columns = list(df.columns)
        self.data_tree['columns'] = columns
        
        # Configure columns
        for col in columns:
            self.data_tree.heading(col, text=col)
            # Set column width based on content
            max_width = max(len(str(col)), 10)  # Minimum width of 10
            if not df[col].empty:
                # Check a sample of values to estimate width
                sample_values = df[col].head(10).astype(str)
                if not sample_values.empty:
                    max_content_width = max(len(str(val)) for val in sample_values)
                    max_width = max(max_width, min(max_content_width, 30))  # Cap at 30 characters
            
            self.data_tree.column(col, width=max_width * 8, minwidth=80)  # Approximate pixel width
        
        # Insert data (limit to first 3000 rows for performance)
        display_rows = min(len(df), 3000)
        for idx, (_, row) in enumerate(df.head(display_rows).iterrows()):
            values = []
            for col in columns:
                val = row[col]
                # Handle different data types and truncate long strings
                try:
                    # Check if value is None, NaN, or similar scalar null values
                    if val is None or (hasattr(val, 'size') and val.size == 1 and pd.isna(val)):
                        display_val = ""
                    # Handle arrays, lists, and other complex types
                    elif hasattr(val, '__iter__') and not isinstance(val, (str, bytes)):
                        # It's an iterable (list, array, etc.) but not a string
                        if hasattr(val, 'shape') and len(val.shape) > 0:
                            # It's a numpy array or similar
                            if val.size == 0:
                                display_val = "[]"
                            else:
                                display_val = str(val.tolist()) if hasattr(val, 'tolist') else str(list(val))
                        else:
                            display_val = str(list(val))
                    else:
                        # Regular scalar value
                        if pd.isna(val):
                            display_val = ""
                        else:
                            display_val = str(val)
                    
                    # Truncate very long values
                    if len(display_val) > 100:
                        display_val = display_val[:97] + "..."
                        
                except Exception as e:
                    # Fallback for any unexpected data types
                    display_val = f"<Error displaying value: {type(val).__name__}>"
                
                values.append(display_val)
            
            self.data_tree.insert('', 'end', values=values)
        
        # Update info if we're showing a subset
        if len(df) > 3000:
            current_info = self.df_info_var.get()
            self.df_info_var.set(f"{current_info} (showing first 3000 rows)")
    
    def clear_data_display(self):
        """Clear the data display"""
        # Clear all items from treeview
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        # Clear column configuration
        self.data_tree['columns'] = ()

    def export_selected_dataframe(self):
        """Export the currently selected dataframe to CSV"""
        selected_name = self.df_selection_var.get()
        if not selected_name or not hasattr(self, 'dataframe_mapping'):
            messagebox.showwarning("No Selection", "Please select a DataFrame to export.")
            return
        
        if selected_name not in self.dataframe_mapping:
            messagebox.showerror("Error", "Selected DataFrame is no longer available.")
            return
        
        dataframe_key = self.dataframe_mapping[selected_name]
        df = self.get_dataframe_by_key(dataframe_key)
        
        if df is None:
            messagebox.showerror("Error", "Selected DataFrame could not be loaded.")
            return
        
        if df.empty:
            messagebox.showwarning("Empty DataFrame", "The selected DataFrame is empty.")
            return
        
        # Open file save dialog
        default_filename = selected_name.replace(" ", "_").replace("(", "").replace(")", "").lower() + ".csv"
        
        file_path = filedialog.asksaveasfilename(
            parent=self.root,
            title=f"Save {selected_name} as CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=default_filename  # Changed from initialvalue to initialfile
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            # Export to CSV
            df.to_csv(file_path, index=False)
            
            # Log success
            self.log_status(f"✓ Exported {selected_name} to {file_path}")
            self.log_status(f"Exported {len(df)} rows and {len(df.columns)} columns")
            
            # Show success message
            messagebox.showinfo("Export Successful", 
                            f"Successfully exported {selected_name} to:\n{file_path}\n\n"
                            f"Rows: {len(df)}\nColumns: {len(df.columns)}")
            
        except Exception as e:
            error_msg = f"Failed to export {selected_name}:\n{str(e)}"
            self.log_status(f"✗ {error_msg}")
            messagebox.showerror("Export Error", error_msg)
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = SessionCreatorApp()
    app.run()
