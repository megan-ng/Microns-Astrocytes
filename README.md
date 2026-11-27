# MICrONS Astrocytes

A Python toolkit for working with astrocyte data from the MICrONS dataset.

This repository contains scripts and reusable functions for retrieving, processing, and analyzing MICrONS electron microscopy (EM) data and astrocyte segmentations. It is intended to support the extraction of image subvolumes, generation of segmentation masks, and downstream analysis of astrocyte morphology and relationships with other cells.

## Repository Structure

```text
Microns-Astrocytes/
│
├── functions/          # Reusable Python modules
│
├── scripts/            # Scripts for running analyses and workflows
│
├── requirements.txt    # Python dependencies
├── .gitignore          # Files and directories excluded from Git
└── README.md           # Project documentation
```

## Installation

### 1. Clone the repository

Open a terminal and navigate to the directory where you want to store the project.

```bash
git clone https://github.com/megan-ng/Microns-Astrocytes.git
cd Microns-Astrocytes
```

### 2. Create a virtual environment

A virtual environment keeps this project's Python packages separate from other Python installations and projects.

Create the environment:

```bash
python3 -m venv venv
```

Activate it:

**macOS / Linux**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\activate
```

Once activated, your terminal should display `(venv)` before the command prompt.

### 3. Install dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

### 4. Install the project

Install the repository in editable mode:

```bash
python -m pip install -e .
```

This allows you to import modules from `functions/` when running scripts, without needing to modify your Python path manually.

## Running Scripts

Activate the virtual environment before running any scripts:

```bash
source venv/bin/activate
```

Scripts can then be run from the repository root. For example:

```bash
python scripts/example_script.py
```