import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import pandas as pd
import session_organizer
from sentence_transformers import SentenceTransformer
import threading
import os
import pickle
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

class ProgressCapture(io.StringIO):
    """Capture progress output and update progress bar."""
    def __init__(self, log_callback, progress_callback):
        super().__init__()
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        
    def write(self, text):
        if text.strip():
            text_stripped = text.strip()
            
            # Check if this is a progress bar update
            if self._is_progress_bar_update(text_stripped):
                # Extract progress percentage and update progress bar
                progress_percent = self._extract_progress_percentage(text_stripped)
                if progress_percent is not None:
                    self.progress_callback(progress_percent)
                # Don't log progress bars to keep output clean
            else:
                # Log non-progress messages normally
                self.log_callback(text_stripped)
        
        return super().write(text)
    
    def _is_progress_bar_update(self, text):
        """Check if the text is a progress bar update."""
        progress_indicators = [
            "Batches:",
            "%|",
            "it/s",
            "[00:",
            "Downloading",
        ]
        return any(indicator in text for indicator in progress_indicators)
    
    def _extract_progress_percentage(self, text):
        """Extract percentage from progress bar text."""
        try:
            # Look for patterns like "Batches:  47%|" or "47%|"
            if "%" in text and "|" in text:
                # Find the percentage value before the | symbol
                percent_part = text.split("%|")[0]
                # Extract the last number before %
                import re
                numbers = re.findall(r'\d+', percent_part)
                if numbers:
                    return int(numbers[-1])
        except (ValueError, IndexError):
            pass
        return None

class SessionCreatorApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # --- Window Configuration ---
        self.title("Presentation Session Organizer")
        self.geometry("750x850")
        self.resizable(True, True)

        # --- Style Configuration ---
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        self.style.configure('TLabel', padding=5)
        self.style.configure('TButton', padding=5)
        self.style.configure('TEntry', padding=5)

        # --- Variables ---
        self.filepath_var = tk.StringVar()
        self.llm_choice_var = tk.StringVar(value="online")
        self.api_key_var = tk.StringVar()
        
        # Column selection variables
        self.title_column_var = tk.StringVar()
        self.abstract_column_var = tk.StringVar()
        self.id_column_var = tk.StringVar()
        
        # Embedding model selection
        self.embedding_model_var = tk.StringVar(value="sentence-transformers/all-MiniLM-L6-v2")
        
        # Data storage
        self.df = None
        self.columns_list = []
        self.embedding_model = None
        
        # Session data for save/load
        self.session_data = None

        # --- Main Frame ---
        main_frame = ttk.Frame(self, padding="10 10 10 10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Widget Creation ---
        self._create_widgets(main_frame)
        self._update_api_key_field_state()

    def _create_widgets(self, parent_frame):
        """Creates and lays out all the widgets in the application."""

        # --- File Selection Section ---
        file_frame = ttk.Frame(parent_frame)
        file_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(file_frame, text="Select the presentation Excel file:").pack(anchor='w')
        
        entry_frame = ttk.Frame(file_frame)
        entry_frame.pack(fill=tk.X, expand=True)
        
        file_entry = ttk.Entry(entry_frame, textvariable=self.filepath_var, state='readonly')
        file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=2)
        
        browse_button = ttk.Button(entry_frame, text="Browse...", command=self.browse_for_file)
        browse_button.pack(side=tk.LEFT, padx=(5, 0))

        # --- Column Selection Section (initially hidden) ---
        ttk.Separator(parent_frame).pack(fill=tk.X, pady=10)
        
        self.column_frame = ttk.Frame(parent_frame)
        self.column_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.column_label = ttk.Label(self.column_frame, text="Select Columns:")
        self.column_label.pack(anchor='w')
        
        # Title column selection
        title_frame = ttk.Frame(self.column_frame)
        title_frame.pack(fill=tk.X, pady=2)
        ttk.Label(title_frame, text="Title Column:", width=15).pack(side=tk.LEFT)
        self.title_combo = ttk.Combobox(title_frame, textvariable=self.title_column_var, state='readonly', width=40)
        self.title_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # Abstract column selection
        abstract_frame = ttk.Frame(self.column_frame)
        abstract_frame.pack(fill=tk.X, pady=2)
        ttk.Label(abstract_frame, text="Abstract Column:", width=15).pack(side=tk.LEFT)
        self.abstract_combo = ttk.Combobox(abstract_frame, textvariable=self.abstract_column_var, state='readonly', width=40)
        self.abstract_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # ID column selection
        id_frame = ttk.Frame(self.column_frame)
        id_frame.pack(fill=tk.X, pady=2)
        ttk.Label(id_frame, text="ID Column:", width=15).pack(side=tk.LEFT)
        self.id_combo = ttk.Combobox(id_frame, textvariable=self.id_column_var, state='readonly', width=40)
        self.id_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # Preview button
        self.preview_button = ttk.Button(self.column_frame, text="Preview Selected Columns", command=self.preview_columns)
        self.preview_button.pack(pady=5)
        
        # Initially hide column selection
        self._hide_column_selection()

        # --- Embedding Model Selection ---
        ttk.Separator(parent_frame).pack(fill=tk.X, pady=10)
        
        embedding_frame = ttk.Frame(parent_frame)
        embedding_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(embedding_frame, text="Select Embedding Model:").pack(anchor='w')
        
        embedding_selection_frame = ttk.Frame(embedding_frame)
        embedding_selection_frame.pack(fill=tk.X, pady=2)
        
        self.embedding_combo = ttk.Combobox(
            embedding_selection_frame, 
            textvariable=self.embedding_model_var, 
            values=[
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/paraphrase-MiniLM-L6-v2",
                "jxm/cde-small-v1"
            ],
            state='readonly',
            width=50
        )
        self.embedding_combo.pack(side=tk.LEFT, padx=(0, 5))
        
        load_model_button = ttk.Button(embedding_selection_frame, text="Load Model", command=self.load_embedding_model)
        load_model_button.pack(side=tk.LEFT)

        # --- Run Session Creation Button ---
        ttk.Separator(parent_frame).pack(fill=tk.X, pady=10)
        
        run_frame = ttk.Frame(parent_frame)
        run_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.run_button = ttk.Button(run_frame, text="Run Session Creation", command=self.run_process)
        self.run_button.pack(anchor='w')

        # --- LLM Choice Section ---
        ttk.Separator(parent_frame).pack(fill=tk.X, pady=10)
        
        llm_frame = ttk.Frame(parent_frame)
        llm_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(llm_frame, text="Choose LLM Method for Title Generation:").pack(anchor='w')

        # Radio buttons and API Key field
        radio_online = ttk.Radiobutton(
            llm_frame, 
            text="Online (Gemini)", 
            variable=self.llm_choice_var, 
            value="online", 
            command=self._update_api_key_field_state
        )
        radio_online.pack(anchor='w', padx=10)

        # Frame to hold the API key label and entry
        self.api_key_frame = ttk.Frame(llm_frame, padding=(20, 0, 0, 0))
        self.api_key_frame.pack(fill=tk.X)
        ttk.Label(self.api_key_frame, text="API Key:").pack(side=tk.LEFT)
        self.api_key_entry = ttk.Entry(self.api_key_frame, textvariable=self.api_key_var, show="*")
        self.api_key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        radio_local = ttk.Radiobutton(
            llm_frame, 
            text="Local (Ollama)", 
            variable=self.llm_choice_var, 
            value="local", 
            command=self._update_api_key_field_state
        )
        radio_local.pack(anchor='w', padx=10)

        # --- Save/Load Section ---
        ttk.Separator(parent_frame).pack(fill=tk.X, pady=10)
        
        saveload_frame = ttk.Frame(parent_frame)
        saveload_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(saveload_frame, text="Session Data Management:").pack(anchor='w')
        
        saveload_buttons = ttk.Frame(saveload_frame)
        saveload_buttons.pack(fill=tk.X, pady=2)
        
        self.save_button = ttk.Button(saveload_buttons, text="Save Session Data", command=self.save_session_data, state='disabled')
        self.save_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.load_button = ttk.Button(saveload_buttons, text="Load Session Data", command=self.load_session_data)
        self.load_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.generate_titles_button = ttk.Button(saveload_buttons, text="Generate Titles Only", command=self.generate_titles_only, state='disabled')
        self.generate_titles_button.pack(side=tk.LEFT)

        # --- Exit Button ---
        ttk.Separator(parent_frame).pack(fill=tk.X, pady=10)

        exit_button = ttk.Button(parent_frame, text="Exit", command=self.destroy)
        exit_button.pack(anchor='w')
        
        # --- Progress Section ---
        progress_frame = ttk.Frame(parent_frame)
        progress_frame.pack(fill=tk.X, pady=(5, 5))
        
        # Main progress bar for overall process
        ttk.Label(progress_frame, text="Overall Progress:").pack(anchor='w')
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        # Sub-progress bar for current step (like embeddings)
        ttk.Label(progress_frame, text="Current Step:").pack(anchor='w')
        self.sub_progress_var = tk.DoubleVar()
        self.sub_progress_bar = ttk.Progressbar(progress_frame, variable=self.sub_progress_var, mode='determinate')
        self.sub_progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        self.progress_label = ttk.Label(progress_frame, text="Ready")
        self.progress_label.pack(anchor='w')
        
        # --- Output/Progress Area ---
        ttk.Label(parent_frame, text="Progress Log:").pack(anchor='w')
        self.output_text = scrolledtext.ScrolledText(parent_frame, height=10, state='disabled', wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def _hide_column_selection(self):
        """Hide the column selection widgets."""
        for widget in self.column_frame.winfo_children():
            widget.pack_forget()

    def _show_column_selection(self):
        """Show the column selection widgets."""
        self.column_label.pack(anchor='w')
        for widget in self.column_frame.winfo_children():
            if widget != self.column_label:
                widget.pack(fill=tk.X, pady=2)
        self.preview_button.pack(pady=5)

    def browse_for_file(self):
        """Opens a file dialog to select an Excel file."""
        filepath = filedialog.askopenfilename(
            title="Select Presentation File",
            filetypes=(("Excel Files", "*.xlsx"), ("Excel Files", "*.xls"), ("All files", "*.*"))
        )
        if filepath:
            self.filepath_var.set(filepath)
            self.load_excel_file(filepath)

    def load_excel_file(self, filepath):
        """Load the Excel file and populate column selection dropdowns."""
        try:
            self.df = pd.read_excel(filepath)
            self.columns_list = list(self.df.columns)
            
            self.log_message(f"File loaded successfully!")
            self.log_message(f"Found {len(self.df)} rows and {len(self.df.columns)} columns.")
            self.log_message(f"Columns: {', '.join(self.columns_list[:5])}{'...' if len(self.columns_list) > 5 else ''}")
            
            # Update comboboxes with column names
            self.title_combo['values'] = self.columns_list
            self.abstract_combo['values'] = self.columns_list
            self.id_combo['values'] = self.columns_list
            
            # Show column selection section
            self._show_column_selection()
            
        except Exception as e:
            self.log_message(f"Error loading file: {str(e)}")
            self._hide_column_selection()

    def preview_columns(self):
        """Preview the selected columns."""
        if self.df is None:
            self.log_message("Error: No file loaded.")
            return
            
        title_col = self.title_column_var.get()
        abstract_col = self.abstract_column_var.get()
        id_col = self.id_column_var.get()
        
        if not all([title_col, abstract_col, id_col]):
            self.log_message("Error: Please select all three columns.")
            return
            
        try:
            preview_df = self.df[[title_col, abstract_col, id_col]]
            self.log_message("\n--- Selected Columns Preview ---")
            self.log_message(f"Title Column: {title_col}")
            self.log_message(f"Abstract Column: {abstract_col}")
            self.log_message(f"ID Column: {id_col}")
            self.log_message("\nFirst 3 rows:")
            
            for idx, row in preview_df.head(3).iterrows():
                self.log_message(f"\nRow {idx}:")
                self.log_message(f"  Title: {str(row[title_col])[:100]}...")
                self.log_message(f"  Abstract: {str(row[abstract_col])[:100]}...")
                self.log_message(f"  ID: {row[id_col]}")
                
        except Exception as e:
            self.log_message(f"Error creating preview: {str(e)}")

    def _update_api_key_field_state(self):
        """Enables or disables the API key entry field based on the radio button selection."""
        if self.llm_choice_var.get() == "online":
            self.api_key_entry.config(state='normal')
            for child in self.api_key_frame.winfo_children():
                child.config(state='normal')
        else:
            self.api_key_entry.config(state='disabled')
            for child in self.api_key_frame.winfo_children():
                child.config(state='disabled')
            self.api_key_var.set("")

    def update_progress(self, value, text=""):
        """Update main progress bar and label."""
        self.progress_var.set(value)
        if text:
            self.progress_label.config(text=text)
        self.update_idletasks()

    def update_sub_progress(self, value):
        """Update sub-progress bar for current step."""
        self.sub_progress_var.set(value)
        self.update_idletasks()

    def reset_sub_progress(self):
        """Reset sub-progress bar."""
        self.sub_progress_var.set(0)
        self.update_idletasks()

    def log_message(self, message):
        """Prints a message to the output text area."""
        self.output_text.config(state='normal')
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.config(state='disabled')
        self.output_text.see(tk.END)
        self.update_idletasks()

    def update_last_line(self, message):
        """Update the last line in the output text area (for progress bars)."""
        self.output_text.config(state='normal')
        
        # Get current content and split into lines
        current_content = self.output_text.get("1.0", tk.END)
        lines = current_content.split('\n')
        
        # If there are lines and the last line is not empty, replace it
        if len(lines) > 1:  # Always has at least one empty line at the end
            # Remove the last empty line and update the previous line
            lines = lines[:-1]  # Remove the last empty element
            if lines:
                lines[-1] = message
            else:
                lines = [message]
        else:
            lines = [message]
        
        # Replace content
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", '\n'.join(lines) + '\n')
        
        self.output_text.config(state='disabled')
        self.output_text.see(tk.END)
        self.update_idletasks()

    def load_embedding_model(self):
        """Load the selected embedding model."""
        model_name = self.embedding_model_var.get()
        if not model_name:
            self.log_message("Error: Please select an embedding model.")
            return
        
        def load_model():
            try:
                self.update_progress(0, "Loading embedding model...")
                self.reset_sub_progress()
                self.log_message(f"Loading embedding model: {model_name}")
                self.log_message("This may take a few minutes for the first time...")
                
                # Capture progress from SentenceTransformer with progress bar updates
                progress_capture = ProgressCapture(self.log_message, self.update_sub_progress)
                with redirect_stdout(progress_capture), redirect_stderr(progress_capture):
                    self.embedding_model = SentenceTransformer(model_name, trust_remote_code=True)
                
                # Log model information
                if hasattr(self.embedding_model, 'model_card_data') and self.embedding_model.model_card_data:
                    base_model = getattr(self.embedding_model.model_card_data, 'base_model', 'Unknown')
                    self.log_message(f"✓ Model loaded successfully: {base_model}")
                else:
                    self.log_message(f"✓ Model loaded successfully: {model_name}")
                
                self.update_progress(100, "Model loaded successfully")
                self.update_sub_progress(100)
                    
            except Exception as e:
                self.log_message(f"✗ Error loading model: {str(e)}")
                self.embedding_model = None
                self.update_progress(0, "Model loading failed")
                self.reset_sub_progress()
        
        # Run model loading in a separate thread
        thread = threading.Thread(target=load_model)
        thread.daemon = True
        thread.start()

    def save_session_data(self):
        """Save current session data to a pickle file."""
        if self.session_data is None:
            messagebox.showwarning("No Data", "No session data to save.")
            return
            
        filepath = filedialog.asksaveasfilename(
            title="Save Session Data",
            defaultextension=".pkl",
            filetypes=(("Pickle Files", "*.pkl"), ("All files", "*.*"))
        )
        
        if filepath:
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(self.session_data, f)
                self.log_message(f"✓ Session data saved to: {filepath}")
            except Exception as e:
                self.log_message(f"✗ Error saving session data: {str(e)}")
                messagebox.showerror("Save Error", f"Failed to save session data: {str(e)}")

    def load_session_data(self):
        """Load session data from a pickle file."""
        filepath = filedialog.askopenfilename(
            title="Load Session Data",
            filetypes=(("Pickle Files", "*.pkl"), ("All files", "*.*"))
        )
        
        if filepath:
            try:
                with open(filepath, 'rb') as f:
                    self.session_data = pickle.load(f)
                
                self.log_message(f"✓ Session data loaded from: {filepath}")
                self.log_message(f"✓ Loaded {len(self.session_data['df_sessions'])} sessions")
                self.log_message(f"✓ Loaded {len(self.session_data['df'])} presentations")
                
                # Enable generate titles button
                self.generate_titles_button.config(state='normal')
                
            except Exception as e:
                self.log_message(f"✗ Error loading session data: {str(e)}")
                messagebox.showerror("Load Error", f"Failed to load session data: {str(e)}")

    def generate_titles_only(self):
        """Generate titles and keywords for loaded session data."""
        if self.session_data is None:
            messagebox.showwarning("No Data", "No session data loaded.")
            return
            
        llm_choice = self.llm_choice_var.get()
        api_key = self.api_key_var.get()
        
        # Validate LLM configuration
        if llm_choice == 'online':
            if not api_key:
                self.log_message("ERROR: API Key is required for online (Gemini) method.")
                messagebox.showerror("API Key Required", "Please enter your Gemini API key.")
                return
            model_name = "gemini-2.0-flash"
            self.log_message("Using Online LLM (Gemini)...")
        else:
            # Check if Ollama is accessible
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code != 200:
                    raise ConnectionError("Ollama server not responding")
                self.log_message("✓ Ollama server is accessible")
            except Exception as e:
                self.log_message(f"✗ Cannot connect to Ollama: {e}")
                messagebox.showerror("Ollama Not Available", "Cannot connect to Ollama server. Please make sure Ollama is running by executing 'ollama serve' in a terminal.")
                return
            
            model_name = "ollama:llama3.2:latest"
            self.log_message("Using Local LLM (Ollama)...")
        
        def generate_titles():
            try:
                self.update_progress(0, "Generating session titles...")
                self.log_message("Generating session titles and keywords...")
                
                df_sessions_with_titles = session_organizer.generate_session_titles_and_keywords(
                    self.session_data['df_sessions'], 
                    self.session_data['df'], 
                    self.session_data['topic_column'], 
                    model_name=model_name
                )
                
                # Update session data
                self.session_data['df_sessions'] = df_sessions_with_titles
                
                # Save results
                output_dir = os.path.dirname(self.session_data['original_filepath'])
                base_name = os.path.splitext(os.path.basename(self.session_data['original_filepath']))[0]
                
                sessions_file = os.path.join(output_dir, f"{base_name}_sessions_with_titles.xlsx")
                df_sessions_with_titles.to_excel(sessions_file, index=False)
                
                self.log_message(f"✓ Saved sessions with titles to: {sessions_file}")
                self.update_progress(100, "Title generation complete")
                
            except Exception as e:
                self.log_message(f"✗ Error generating titles: {str(e)}")
                self.update_progress(0, "Title generation failed")
        
        thread = threading.Thread(target=generate_titles)
        thread.daemon = True
        thread.start()

    def run_process(self):
        """The main function to call when the 'Run' button is clicked."""
        filepath = self.filepath_var.get()
        
        title_col = self.title_column_var.get()
        abstract_col = self.abstract_column_var.get()
        id_col = self.id_column_var.get()

        # Validation - only check for file, columns, and embedding model
        if not filepath:
            self.log_message("ERROR: Please select a file first.")
            messagebox.showerror("File Required", "Please select an Excel file.")
            return
            
        if not all([title_col, abstract_col, id_col]):
            self.log_message("ERROR: Please select all required columns.")
            messagebox.showerror("Columns Required", "Please select all three required columns.")
            return
            
        if self.embedding_model is None:
            self.log_message("ERROR: Please load an embedding model first.")
            messagebox.showerror("Model Required", "Please load an embedding model first.")
            return
        
        # Disable buttons during processing
        self.run_button.config(state='disabled')
        self.save_button.config(state='disabled')
        self.generate_titles_button.config(state='disabled')
        
        def process_sessions():
            try:
                self.log_message("=" * 50)
                self.log_message("STARTING SESSION CREATION PROCESS")
                self.log_message("=" * 50)
                
                # Step 1: Load presentations
                self.update_progress(10, "Loading presentations...")
                self.reset_sub_progress()
                self.log_message("Step 1: Loading presentations...")
                df, title_column, abstract_column, abstract_id_column, topic_column = session_organizer.load_presentations(
                    filepath,
                    Title_name=title_col,
                    Abstract_name=abstract_col,
                    Abstract_ID_name=id_col
                )
                self.log_message(f"✓ Loaded {len(df)} presentations")
                
                # Step 2: Create embeddings with progress bar updates
                self.update_progress(20, "Creating embeddings...")
                self.reset_sub_progress()
                self.log_message("Step 2: Creating embeddings...")
                
                progress_capture = ProgressCapture(self.log_message, self.update_sub_progress)
                with redirect_stdout(progress_capture), redirect_stderr(progress_capture):
                    df_presentation_embeddings = session_organizer.embed_documents(df, topic_column, self.embedding_model)
                
                self.log_message(f"✓ Created embeddings with shape: {df_presentation_embeddings.shape}")
                self.update_sub_progress(100)
                
                # Step 3: Calculate similarity matrix
                self.update_progress(40, "Calculating similarity matrix...")
                self.reset_sub_progress()
                self.log_message("Step 3: Calculating similarity matrix...")
                df_presentation_similarities = session_organizer.calculate_similarity_matrix(df_presentation_embeddings, df, self.embedding_model)
                self.log_message(f"✓ Calculated similarity matrix with shape: {df_presentation_similarities.shape}")
                self.update_sub_progress(100)
                
                # Step 4: Remove duplicates
                self.update_progress(50, "Removing duplicates...")
                self.reset_sub_progress()
                self.log_message("Step 4: Removing near-duplicates...")
                df, df_presentation_similarities, df_presentation_embeddings = session_organizer.remove_duplicates(
                    df, df_presentation_similarities, df_presentation_embeddings, threshold=0.99
                )
                self.log_message(f"✓ Final dataset: {len(df)} presentations")
                self.update_sub_progress(100)
                
                # Step 5: Create sessions
                self.update_progress(60, "Creating sessions...")
                self.reset_sub_progress()
                self.log_message("Step 5: Creating sessions...")
                df, df_sessions, labels, metadata = session_organizer.create_sessions(
                    df, df_presentation_similarities, df_presentation_embeddings, 
                    max_sessions=100, min_session_size=8, tree_merge_stop=1, 
                    cluster_column_name="Session Code"
                )
                self.log_message(f"✓ Created {metadata['n_clusters']} sessions")
                self.log_message(f"  - Assigned presentations: {metadata['n_assigned_items']}")
                self.log_message(f"  - Unassigned presentations: {metadata['n_unassigned_items']}")
                self.update_sub_progress(100)
                
                # Step 6: Analyze sessions
                self.update_progress(70, "Analyzing sessions...")
                self.reset_sub_progress()
                self.log_message("Step 6: Analyzing sessions...")
                df_sessions['session_coherence'] = session_organizer.calculate_avg_similarity(df_sessions, df_presentation_similarities.values)
                df_sessions['session_distinctiveness'] = session_organizer.calculate_silhouette_scores(df_sessions, df_presentation_embeddings.values, labels)
                df['presentation_session_fit'] = session_organizer.calculate_document_similarities(df_presentation_similarities.values, labels)
                self.log_message("✓ Session analysis complete")
                self.update_sub_progress(100)
                
                # Save intermediate results
                self.update_progress(80, "Saving intermediate results...")
                self.reset_sub_progress()
                self.session_data = {
                    'df': df,
                    'df_sessions': df_sessions,
                    'df_presentation_embeddings': df_presentation_embeddings,
                    'df_presentation_similarities': df_presentation_similarities,
                    'labels': labels,
                    'metadata': metadata,
                    'topic_column': topic_column,
                    'original_filepath': filepath
                }
                
                output_dir = os.path.dirname(filepath)
                base_name = os.path.splitext(os.path.basename(filepath))[0]
                
                # Auto-save session data
                session_data_file = os.path.join(output_dir, f"{base_name}_session_data.pkl")
                with open(session_data_file, 'wb') as f:
                    pickle.dump(self.session_data, f)
                self.log_message(f"✓ Saved session data to: {session_data_file}")
                
                # Save Excel files
                sessions_file = os.path.join(output_dir, f"{base_name}_sessions.xlsx")
                presentations_file = os.path.join(output_dir, f"{base_name}_presentations.xlsx")
                
                df_sessions.to_excel(sessions_file, index=False)
                df.to_excel(presentations_file, index=False)
                
                self.log_message(f"✓ Saved sessions to: {sessions_file}")
                self.log_message(f"✓ Saved presentations to: {presentations_file}")
                self.update_sub_progress(100)
                
                # Enable save and generate titles buttons
                self.save_button.config(state='normal')
                self.generate_titles_button.config(state='normal')
                
                self.update_progress(100, "Session creation complete!")
                self.log_message("=" * 50)
                self.log_message("SESSION CREATION COMPLETE!")
                self.log_message("You can now configure LLM settings and generate titles.")
                self.log_message("=" * 50)
                
            except Exception as e:
                self.log_message(f"✗ ERROR: {str(e)}")
                import traceback
                self.log_message(f"Full error: {traceback.format_exc()}")
                self.update_progress(0, "Process failed")
                self.reset_sub_progress()
            finally:
                # Re-enable the run button
                self.run_button.config(state='normal')
        
        # Run processing in a separate thread
        thread = threading.Thread(target=process_sessions)
        thread.daemon = True
        thread.start()


if __name__ == "__main__":
    app = SessionCreatorApp()
    app.mainloop()

