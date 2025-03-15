import os
import zipfile
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import shutil
import argparse
import uuid

def process_zip_file(args):
    zip_path, output_dir = args
    try:
        # Create a unique temp directory for this zip file
        temp_dir = output_dir / f"temp_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get all .ttf files from the zip
            ttf_files = [f for f in zip_ref.namelist() if f.lower().endswith('.ttf')]
            
            # Extract each .ttf file
            for ttf_file in ttf_files:
                try:
                    # Extract with a sanitized filename (keep only the basename)
                    safe_name = os.path.basename(ttf_file)
                    # If file already exists, add a number to the filename
                    counter = 1
                    base, ext = os.path.splitext(safe_name)
                    while (output_dir / safe_name).exists():
                        safe_name = f"{base}_{counter}{ext}"
                        counter += 1
                    
                    try:
                        # Extract the file
                        zip_ref.extract(ttf_file, temp_dir)
                        # Create any necessary parent directories
                        (output_dir / safe_name).parent.mkdir(parents=True, exist_ok=True)
                        # Move it to the main output directory with the safe name
                        src_path = temp_dir / ttf_file
                        dst_path = output_dir / safe_name
                        if src_path.exists():  # Only try to move if the file exists
                            shutil.move(str(src_path), str(dst_path))
                    except FileNotFoundError as e:
                        print(f"Warning: Could not find {ttf_file} in {zip_path}")
                    except Exception as e:
                        print(f"Error extracting {ttf_file} from {zip_path}: {e}")
                except Exception as e:
                    print(f"Error processing {ttf_file} from {zip_path}: {e}")
            
            # Clean up temp directory for this zip
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Could not clean up temp directory for {zip_path}: {e}")
                
        return True
    except Exception as e:
        print(f"Error processing zip file {zip_path}: {e}")
        # Try to clean up temp directory even if we failed
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        except:
            pass
        return False

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Extract TTF files from ZIP archives.')
    parser.add_argument('--input', '-i', type=str, default="./fonts", 
                        help='Directory containing ZIP files (default: downloads folder in script directory)')
    parser.add_argument('--output', '-o', type=str, default="./fonts_unpacked",
                        help='Directory to extract fonts to (default: extracted_fonts folder in script directory)')
    args = parser.parse_args()
    
    # Setup directories
    current_dir = Path(__file__).parent
    zip_dir = Path(args.input) if args.input else current_dir / "downloads"
    output_dir = Path(args.output) if args.output else current_dir / "extracted_fonts"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of zip files
    zip_files = list(zip_dir.glob("*.zip"))
    
    if not zip_files:
        print(f"No zip files found in the directory: {zip_dir}")
        return
    
    # Prepare arguments for multiprocessing
    args = [(zip_file, output_dir) for zip_file in zip_files]
    
    # Use number of CPU cores minus 1 (or at least 1)
    num_processes = max(1, mp.cpu_count() - 1)
    
    # Process zip files in parallel with progress bar
    with mp.Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_zip_file, args),
            total=len(zip_files),
            desc="Extracting fonts"
        ))
    
    # Print summary
    successful = sum(1 for r in results if r)
    failed = len(results) - successful
    print(f"\nExtraction complete!")
    print(f"Successfully processed: {successful} zip files")
    print(f"Failed to process: {failed} zip files")
    
    # Clean up any remaining temp directories (just in case)
    for temp_dir in output_dir.glob("temp_*"):
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temp directory {temp_dir}: {e}")

if __name__ == "__main__":
    main()
