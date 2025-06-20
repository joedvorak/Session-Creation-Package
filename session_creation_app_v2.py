import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import pandas as pd
import os
import requests
from sentence_transformers import SentenceTransformer
import session_organizer

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
        
        save_state_btn = ttk.Button(state_buttons_frame, text="Save Process State")
        save_state_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        load_state_btn = ttk.Button(state_buttons_frame, text="Load Process State")
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
        
        # Bottom section - Progress and Status (always visible)
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.create_status_section(bottom_frame)
        
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
                                         "jxm/cde-small-v1"], state='readonly')
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
        
        save_btn = ttk.Button(save_load_frame, text="Save Analysis")
        save_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        
        load_btn = ttk.Button(save_load_frame, text="Load Analysis")
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
        """Add a message to the status log"""
        if self.status_text:
            self.status_text.config(state='normal')
            self.status_text.insert(tk.END, f"{message}\n")
            self.status_text.see(tk.END)  # Scroll to bottom
            self.status_text.config(state='disabled')
            self.root.update_idletasks()  # Update GUI immediately

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
        
        assign_btn = ttk.Button(committees_frame, text="Assign Committees")
        assign_btn.pack()
        
        # Export Results
        export_frame = ttk.LabelFrame(parent, text="Save Results", padding="10")
        export_frame.pack(fill=tk.X, padx=5, pady=5)
        
        export_btn = ttk.Button(export_frame, text="Export Spreadsheets")
        export_btn.pack()
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
            
            # Set environment variable for API key
            import os
            if selected_model == "gemini-2.0-flash":
                os.environ["GEMINI_API_KEY"] = api_key
            # Add other API key mappings as needed
        
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
            
            # Call the session_organizer function
            df_sessions_with_titles = session_organizer.generate_session_titles_and_keywords(
                df_sessions=self.df_sessions,
                df_presentations=df_presentations,
                topic_column=topic_column,
                model_name=model_name,
                prompt_template=None  # Use default prompt
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
            df_result, df_sessions, labels, metadata = session_organizer.create_sessions_w_hybrid(
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
                final_session_title_column="Final Session Title"
            )
            
            # Store results
            self.normal_presentations_data['processed_dataframe'] = df_result  # Update with session assignments
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

    def run(self):
        self.root.mainloop()

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

if __name__ == "__main__":
    app = SessionCreatorApp()
    app.run()
