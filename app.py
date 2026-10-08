import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

import mne
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Force MNE to use qt backend
mne.viz.set_browser_backend("qt")

TARGET_CHANNELS = ["F3", "F4", "C3", "Cz", "C4", "P3", "P4"]

# Preset annotation options for clinical / research scoring
PRESET_ANNOTATIONS = ["artifact", "IBI", "awake", "activeSleep", "indeterminateSleep", "quietSleep", "QSburst"]


class EEGLoaderThread(QThread):
    """Worker thread that reads EDF signals and applies filters off the main GUI thread."""
    finished = Signal(object, object, list, str)  # raw_original, raw_processed, found_channels, error_msg

    def __init__(self, file_path, hp_freq, lp_freq, use_notch, ref_mode):
        super().__init__()
        self.file_path = file_path
        self.hp_freq = hp_freq
        self.lp_freq = lp_freq
        self.use_notch = use_notch
        self.ref_mode = ref_mode

    def run(self):
        try:
            reader = (
                mne.io.read_raw_edf
                if self.file_path.suffix.lower() in [".edf", ".bdf"]
                else mne.io.read_raw_fif
            )
            loaded_raw = reader(self.file_path, preload=True, verbose=False)

            file_channels = loaded_raw.ch_names
            found_channels = []
            for target in TARGET_CHANNELS:
                for ch in file_channels:
                    if ch.strip().upper() == target.upper():
                        found_channels.append(ch)
                        break

            if not found_channels:
                self.finished.emit(
                    None,
                    None,
                    [],
                    f"None of required channels {TARGET_CHANNELS} found in {self.file_path.name}."
                )
                return

            loaded_raw.pick(found_channels)
            montage = mne.channels.make_standard_montage("standard_1020")
            loaded_raw.set_montage(montage, on_missing="ignore")

            # --- SEED PRESET ANNOTATIONS INTO MNE ---
            # Inject 0-duration dummy annotations at t=0s for presets so MNE populates buttons automatically
            existing_descriptions = set(loaded_raw.annotations.description)
            dummy_onsets = []
            dummy_durations = []
            dummy_descriptions = []

            for preset in PRESET_ANNOTATIONS:
                if preset not in existing_descriptions:
                    dummy_onsets.append(0.0)
                    dummy_durations.append(0.0)
                    dummy_descriptions.append(preset)

            if dummy_descriptions:
                dummy_annots = mne.Annotations(
                    onset=dummy_onsets,
                    duration=dummy_durations,
                    description=dummy_descriptions,
                    orig_time=loaded_raw.annotations.orig_time
                )
                loaded_raw.set_annotations(loaded_raw.annotations + dummy_annots)

            raw_original = loaded_raw
            raw_processed = raw_original.copy()

            # Bandpass Filter
            if self.hp_freq < self.lp_freq:
                raw_processed.filter(l_freq=self.hp_freq, h_freq=self.lp_freq, verbose=False)

            # Notch Filter
            if self.use_notch:
                raw_processed.notch_filter(freqs=np.arange(60, 241, 60), verbose=False)

            # Reference Display Mode
            if self.ref_mode == "Bipolar (Longitudinal Chain)":
                anodes = ["F3", "C3", "F4", "C4"]
                cathodes = ["C3", "P3", "C4", "P4"]
                raw_processed = mne.set_bipolar_reference(
                    raw_processed, anode=anodes, cathode=cathodes, drop_refs=True, verbose=False
                )
            elif self.ref_mode == "Bipolar (Transverse Pairs)":
                anodes = ["F3", "C3", "Cz", "P3"]
                cathodes = ["F4", "Cz", "C4", "P4"]
                raw_processed = mne.set_bipolar_reference(
                    raw_processed, anode=anodes, cathode=cathodes, drop_refs=True, verbose=False
                )
            elif self.ref_mode == "Bipolar (Modified double banana)":
                anodes = ['F3', 'C3', 'F4', 'C4', 'F3', 'C3', 'P3']
                cathodes = ['C3', 'P3', 'C4', 'P4', 'F4', 'C4', 'P4']
                raw_processed = mne.set_bipolar_reference(
                    raw_processed, anode=anodes, cathode=cathodes, drop_refs=True, verbose=False
                )

            self.finished.emit(raw_original, raw_processed, found_channels, "")

        except Exception as e:
            self.finished.emit(None, None, [], str(e))


class EEGViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Batch EEG Viewer & Annotator (7-Channel)")
        self.resize(1380, 800)

        # Dataset & Files State
        self.data_dir = None
        self.metadata_df = None
        self.edf_map = {}  # Dropdown Index -> Path
        self.current_index = -1

        # Current File State
        self.loaded_file_path = None
        self.raw_original = None
        self.raw = None
        self.browser = None
        self.loader_thread = None
        self.progress_dialog = None

        # Tracking annotation sets
        self.original_annotations = None
        self.new_annotations = None

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.setCentralWidget(splitter)

        # Left: EEG Container
        self.eeg_box = QWidget()
        self.eeg_layout = QVBoxLayout(self.eeg_box)
        self.eeg_layout.setContentsMargins(0, 0, 0, 0)

        self.placeholder = QLabel("Select a folder containing EDFs & Metadata to begin")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.eeg_layout.addWidget(self.placeholder)
        splitter.addWidget(self.eeg_box)

        # Right: Sidebar
        sidebar = QWidget()
        side_layout = QVBoxLayout(sidebar)

        # 1. Dataset Folder & Selector Group
        ds_group = QGroupBox("Dataset & File Selector")
        ds_layout = QVBoxLayout(ds_group)

        btn_load_dir = QPushButton("Open EDF Folder")
        btn_load_dir.clicked.connect(self.load_dataset_folder)
        ds_layout.addWidget(btn_load_dir)

        ds_layout.addWidget(QLabel("Select Subject / File:"))
        self.edf_combo = QComboBox()
        self.edf_combo.activated.connect(self.on_edf_combo_activated)
        ds_layout.addWidget(self.edf_combo)

        # Metadata display labels
        self.lbl_subject_info = QLabel("<b>Participant ID:</b> N/A")
        self.lbl_ga_weeks = QLabel("<b>Gestational Age (GA):</b> N/A")
        self.lbl_annot_status = QLabel("<b>Existing Annotations:</b> N/A")
        self.lbl_annot_status.setWordWrap(True)

        ds_layout.addWidget(self.lbl_subject_info)
        ds_layout.addWidget(self.lbl_ga_weeks)
        ds_layout.addWidget(self.lbl_annot_status)

        # Explicit Load Data Button
        self.btn_load_data = QPushButton("Load Data")
        self.btn_load_data.setStyleSheet(
            "background-color: #1976D2; color: white; font-weight: bold; padding: 6px;"
        )
        self.btn_load_data.clicked.connect(self.on_load_data_clicked)
        ds_layout.addWidget(self.btn_load_data)

        side_layout.addWidget(ds_group)

        self.lbl_channels = QLabel("<b>Channels:</b> None loaded")
        self.lbl_channels.setWordWrap(True)
        side_layout.addWidget(self.lbl_channels)

        # 2. Reference Mode Selector
        side_layout.addWidget(QLabel("<b>Reference Display Mode:</b>"))
        self.ref_combo = QComboBox()
        self.ref_combo.addItems([
            "Referential (Single-ended)",
            "Bipolar (Longitudinal Chain)",
            "Bipolar (Transverse Pairs)",
            "Bipolar (Modified double banana)"
        ])
        self.ref_combo.currentIndexChanged.connect(self.apply_pipeline)
        side_layout.addWidget(self.ref_combo)

        # 3. Signal Filtering Group
        filter_group = QGroupBox("Signal Filtering")
        filter_layout = QVBoxLayout(filter_group)

        self.notch_chk = QCheckBox("60 Hz Notch Filter")
        self.notch_chk.setChecked(True)
        self.notch_chk.stateChanged.connect(self.apply_pipeline)
        filter_layout.addWidget(self.notch_chk)

        bp_layout = QHBoxLayout()
        bp_layout.addWidget(QLabel("HP (Hz):"))
        self.hp_spin = QDoubleSpinBox()
        self.hp_spin.setRange(0.1, 100.0)
        self.hp_spin.setValue(1.0)
        self.hp_spin.setSingleStep(0.5)
        bp_layout.addWidget(self.hp_spin)

        bp_layout.addWidget(QLabel("LP (Hz):"))
        self.lp_spin = QDoubleSpinBox()
        self.lp_spin.setRange(1.0, 500.0)
        self.lp_spin.setValue(30.0)
        self.lp_spin.setSingleStep(5.0)
        bp_layout.addWidget(self.lp_spin)

        filter_layout.addLayout(bp_layout)

        btn_bp_apply = QPushButton("Apply Bandpass")
        btn_bp_apply.clicked.connect(self.apply_pipeline)
        filter_layout.addWidget(btn_bp_apply)

        btn_reset_filters = QPushButton("Reset All Filters")
        btn_reset_filters.clicked.connect(self.reset_filters)
        filter_layout.addWidget(btn_reset_filters)

        side_layout.addWidget(filter_group)

        # 4. Annotations Section
        side_layout.addWidget(QLabel("<b>Annotations:</b>"))

        tag_container = QHBoxLayout()
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(PRESET_ANNOTATIONS)
        tag_container.addWidget(self.combo_preset)

        lbl_hint = QLabel(
            "<small><i>"
            "• Press 'a' in plot to annotate regions using preset descriptions<br>"
            "• To annotate: click & drag, or click & advance with keyboard arrows<br>"
            "• To delete an annotation: right-click the highlighted area (cannot be undone)<br>"
            "• When complete: click 'SAVE Annotations'"
            "</i></small>"
        )
        lbl_hint.setWordWrap(True)  # Prevents long lines from cutting off
        lbl_hint.setStyleSheet("color: gray;")
        side_layout.addWidget(lbl_hint)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels([
            "Onset (s)",
            "Duration (s)",
            "Description",
        ])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        side_layout.addWidget(self.table)

        btn_confirm = QPushButton("CONFIRM (Sync Annotations)")
        btn_confirm.setStyleSheet("font-weight: bold; padding: 6px;")
        btn_confirm.clicked.connect(self.sync_annotations)
        side_layout.addWidget(btn_confirm)

        btn_save = QPushButton("SAVE Annotations")
        btn_save.setStyleSheet(
            "background-color: #2e7d32; color: white; font-weight: bold; padding: 8px;"
        )
        btn_save.clicked.connect(self.save_annotations)
        side_layout.addWidget(btn_save)

        splitter.addWidget(sidebar)
        splitter.setSizes([920, 460])

    def load_dataset_folder(self):
        """Loads a folder, parses metadata, and populates selector without loading EDF signal data."""
        if not self.check_unsaved_changes():
            return

        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing EDFs & Metadata")
        if not folder:
            return

        self.data_dir = Path(folder)

        meta_files = [f for f in self.data_dir.iterdir() if f.suffix.lower() in [".csv", ".xlsx", ".xls"]]
        if not meta_files:
            QMessageBox.warning(self, "Metadata Missing", "No CSV or Excel metadata file found in the selected folder.")
            return

        meta_path = meta_files[0]
        try:
            if meta_path.suffix.lower() == ".csv":
                df = pd.read_csv(meta_path)
            else:
                df = pd.read_excel(meta_path)

            req_cols = ["participant_id", "filename", "ga_weeks"]
            missing = [col for col in req_cols if col not in df.columns]
            if missing:
                QMessageBox.critical(self, "Metadata Error", f"Metadata file is missing columns: {missing}")
                return

            self.metadata_df = df

        except Exception as e:
            QMessageBox.critical(self, "Error Reading Metadata", str(e))
            return

        self.edf_combo.blockSignals(True)
        self.edf_combo.clear()
        self.edf_map.clear()

        existing_files_map = {f.stem.lower(): f for f in self.data_dir.iterdir() if f.is_file()}

        idx = 0
        for _, row in self.metadata_df.iterrows():
            fname = str(row["filename"]).strip()
            p_id = str(row["participant_id"]).strip()
            ga = str(row["ga_weeks"]).strip()

            target_path = self.data_dir / fname

            if target_path.exists():
                file_path = target_path
            else:
                file_path = existing_files_map.get(Path(fname).stem.lower())

            if file_path and file_path.exists():
                display_label = f"ID: {p_id} | GA: {ga} weeks"
                self.edf_combo.addItem(display_label)
                self.edf_map[idx] = {
                    "path": file_path,
                    "participant_id": p_id,
                    "ga_weeks": ga
                }
                idx += 1

        self.edf_combo.blockSignals(False)

        if idx > 0:
            self.edf_combo.setCurrentIndex(0)
            self.select_patient_metadata_only(0)
        else:
            QMessageBox.warning(self, "No Files Matched", "None of the files in the metadata exist in this folder.")

    def on_edf_combo_activated(self, target_index):
        """Triggered when selecting a different file in the dropdown without loading signals."""
        if target_index == self.current_index:
            return

        if not self.check_unsaved_changes():
            self.edf_combo.blockSignals(True)
            self.edf_combo.setCurrentIndex(self.current_index)
            self.edf_combo.blockSignals(False)
            return

        self.select_patient_metadata_only(target_index)

    def select_patient_metadata_only(self, index):
        """Updates display labels without reading heavy EDF signal data into memory."""
        if index not in self.edf_map:
            return

        self.current_index = index
        item_data = self.edf_map[index]
        edf_path = item_data["path"]

        self.lbl_subject_info.setText(f"<b>Participant ID:</b> {item_data['participant_id']}")
        self.lbl_ga_weeks.setText(f"<b>Gestational Age (GA):</b> {item_data['ga_weeks']} weeks")

        annotations_dir = self.data_dir / "annotations"
        stem = edf_path.stem
        existing_annots = []

        if annotations_dir.exists():
            existing_annots = list(annotations_dir.glob(f"{stem}_*_annotations_*.csv"))

        if existing_annots:
            filenames_str = ", ".join([f.name for f in existing_annots])
            self.lbl_annot_status.setText(
                f"<b>Existing Annotations:</b> <span style='color: green;'>Found {len(existing_annots)} file(s)</span><br>"
                f"<small><i>({filenames_str})</i></small>"
            )
        else:
            self.lbl_annot_status.setText("<b>Existing Annotations:</b> <span style='color: gray;'>None found</span>")

    def on_load_data_clicked(self):
        """Confirms folder contents and annotation statuses before loading EDF data."""
        if self.current_index not in self.edf_map or self.data_dir is None:
            QMessageBox.warning(self, "Warning", "Please select a valid subject first.")
            return

        total_meta_files = len(self.metadata_df)
        found_eeg_files = list(self.edf_map.values())
        total_found = len(found_eeg_files)

        missing_files = []
        existing_stems = {f["path"].stem.lower() for f in found_eeg_files}
        for _, row in self.metadata_df.iterrows():
            fname = str(row["filename"]).strip()
            if Path(fname).stem.lower() not in existing_stems:
                missing_files.append(fname)

        annotations_dir = self.data_dir / "annotations"
        new_annot_files = []
        if annotations_dir.exists():
            new_annot_files = list(annotations_dir.glob("*_new_annotations_*.csv"))

        selected_file = self.edf_map[self.current_index]["path"].name

        msg_lines = [
            f"<b>Selected File to Load:</b> {selected_file}<br>",
            f"<b>EDF Folder Audit:</b>",
            f"• Found <b>{total_found}</b> of <b>{total_meta_files}</b> expected EDF files in selected folder."
        ]

        if missing_files:
            missing_str = ", ".join(missing_files[:5])
            if len(missing_files) > 5:
                missing_str += f" (+{len(missing_files) - 5} more)"
            msg_lines.append(f"• <span style='color: red;'>Missing Files ({len(missing_files)}):</span> {missing_str}")

        msg_lines.append("<br><b>Annotations Audit:</b>")
        if new_annot_files:
            msg_lines.append(
                f"• Found <span style='color: green;'><b>{len(new_annot_files)}</b> associated 'new annotation' file(s)</span> in /annotations.")
        else:
            msg_lines.append("• <span style='color: gray;'>No 'new annotation' files found in /annotations.</span>")

        msg_lines.append("<br><b>Do you want to proceed with loading this file?</b>")

        reply = QMessageBox.question(
            self,
            "Confirm Data Loading",
            "<br>".join(msg_lines),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )

        if reply == QMessageBox.StandardButton.Yes:
            edf_path = self.edf_map[self.current_index]["path"]
            self.load_eeg_file_async(edf_path)

    def load_eeg_file_async(self, file_path):
        """Starts the background thread and shows an animated progress dialog."""
        self.loaded_file_path = file_path

        self.progress_dialog = QProgressDialog("Reading EDF signals & applying filters...", None, 0, 0, self)
        self.progress_dialog.setWindowTitle("Loading EEG Data")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.show()

        self.loader_thread = EEGLoaderThread(
            file_path,
            self.hp_spin.value(),
            self.lp_spin.value(),
            self.notch_chk.isChecked(),
            self.ref_combo.currentText()
        )
        self.loader_thread.finished.connect(self.on_eeg_loaded)
        self.loader_thread.start()

    def on_eeg_loaded(self, raw_orig, raw_proc, found_channels, error_msg):
        """Callback triggered when background thread completes execution."""
        if self.progress_dialog:
            self.progress_dialog.close()

        if error_msg:
            QMessageBox.critical(self, "Error Loading EEG File", error_msg)
            return

        self.raw_original = raw_orig
        self.raw = raw_proc
        self.original_annotations = raw_orig.annotations.copy()

        self.lbl_channels.setText(
            f"<b>Channels ({len(found_channels)}):</b> {', '.join(found_channels)}"
        )

        self.refresh_browser()

    def add_quick_preset_annotation(self):
        """Directly inserts a 1.0s annotation segment for the selected preset from the sidebar."""
        if self.raw is None:
            QMessageBox.warning(self, "Warning", "Please load an EEG file first.")
            return

        selected_label = self.combo_preset.currentText()
        new_annot = mne.Annotations(
            onset=[0.0],
            duration=[1.0],
            description=[selected_label],
            orig_time=self.raw.annotations.orig_time
        )
        self.raw.set_annotations(self.raw.annotations + new_annot)
        self.refresh_browser()

    def apply_pipeline(self):
        if self.raw_original is None or not self.loaded_file_path:
            return
        self.load_eeg_file_async(self.loaded_file_path)

    def reset_filters(self):
        self.notch_chk.setChecked(True)
        self.hp_spin.setValue(1.0)
        self.lp_spin.setValue(30.0)
        self.ref_combo.setCurrentIndex(0)
        self.apply_pipeline()

    def refresh_browser(self):
        if self.raw is None:
            return

        if self.browser:
            self.eeg_layout.removeWidget(self.browser)
            self.browser.close()
            self.browser.deleteLater()
            self.browser = None

        if self.placeholder:
            self.placeholder.setParent(None)

        self.browser = self.raw.plot(
            show=False,
            block=False,
            show_options=True
        )
        self.eeg_layout.addWidget(self.browser)
        self.sync_annotations()

    def sync_annotations(self):
        """Fetches plot annotations and splits into original vs new, filtering out 0-duration dummies."""
        self.table.setRowCount(0)
        if not self.raw or not self.raw.annotations:
            return

        current_annots = self.raw.annotations

        orig_keys = set()
        if self.original_annotations:
            for ann in self.original_annotations:
                # Ignore 0s dummy placeholders
                if ann["duration"] == 0.0 and ann["onset"] == 0.0 and ann["description"] in PRESET_ANNOTATIONS:
                    continue
                orig_keys.add(
                    (round(ann["onset"], 3), round(ann["duration"], 3), ann["description"])
                )

        new_onsets, new_durations, new_descriptions = [], [], []
        for ann in current_annots:
            # Ignore 0s dummy placeholders
            if ann["duration"] == 0.0 and ann["onset"] == 0.0 and ann["description"] in PRESET_ANNOTATIONS:
                continue

            key = (round(ann["onset"], 3), round(ann["duration"], 3), ann["description"])
            if key not in orig_keys:
                new_onsets.append(ann["onset"])
                new_durations.append(ann["duration"])
                new_descriptions.append(ann["description"])

            # Render valid annotations in table
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(f"{ann['onset']:.3f}"))
            self.table.setItem(row, 1, QTableWidgetItem(f"{ann['duration']:.3f}"))
            self.table.setItem(row, 2, QTableWidgetItem(str(ann['description'])))

        self.new_annotations = mne.Annotations(
            onset=new_onsets,
            duration=new_durations,
            description=new_descriptions,
            orig_time=current_annots.orig_time,
        )

    def has_unsaved_annotations(self) -> bool:
        """Returns True if there are newly added annotations that haven't been saved."""
        if not self.raw:
            return False
        self.sync_annotations()
        return self.new_annotations is not None and len(self.new_annotations) > 0

    def check_unsaved_changes(self) -> bool:
        """Prompts user if unsaved annotations exist."""
        if not self.has_unsaved_annotations():
            return True

        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.setWindowTitle("Unsaved Annotations")
        msg_box.setText("You have unsaved annotations for this file!")
        msg_box.setInformativeText("Would you like to save them before leaving?")

        btn_save = msg_box.addButton("Save Annotations", QMessageBox.ButtonRole.AcceptRole)
        btn_discard = msg_box.addButton("Delete / Discard", QMessageBox.ButtonRole.DestructiveRole)
        btn_cancel = msg_box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)

        msg_box.setDefaultButton(btn_save)
        msg_box.exec()

        clicked = msg_box.clickedButton()

        if clicked == btn_save:
            return self.save_annotations()
        elif clicked == btn_discard:
            return True
        else:
            return False

    def save_annotations(self) -> bool:
        """Saves separate files for original and new annotations strictly into <EDF_Folder>/annotations."""
        if not self.raw or not self.loaded_file_path or self.data_dir is None:
            QMessageBox.warning(self, "Warning", "No EEG file loaded or dataset directory selected.")
            return False

        self.sync_annotations()

        annotations_dir = self.data_dir / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)

        stem = self.loaded_file_path.stem
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        orig_filename = f"{stem}_original_annotations_{date_str}.csv"
        new_filename = f"{stem}_new_annotations_{date_str}.csv"

        orig_path = annotations_dir / orig_filename
        new_path = annotations_dir / new_filename

        try:
            saved_messages = []

            if self.original_annotations and len(self.original_annotations) > 0:
                self.original_annotations.save(orig_path, overwrite=True)
                saved_messages.append(f"Originals: {orig_filename}")
            else:
                saved_messages.append("Originals: None found in raw file")

            if self.new_annotations and len(self.new_annotations) > 0:
                self.new_annotations.save(new_path, overwrite=True)
                saved_messages.append(f"New Additions ({len(self.new_annotations)}): {new_filename}")
            else:
                saved_messages.append("New Additions: None added")

            msg = f"Saved to directory:\n{annotations_dir}\n\n" + "\n".join(saved_messages)
            QMessageBox.information(self, "Annotations Saved", msg)

            self.original_annotations = self.raw.annotations.copy()

            self.select_patient_metadata_only(self.current_index)
            return True

        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))
            return False

    def closeEvent(self, event: QCloseEvent):
        """Intercepts window close/exit to warn about unsaved annotations."""
        if self.check_unsaved_changes():
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = EEGViewer()
    win.show()
    sys.exit(app.exec())