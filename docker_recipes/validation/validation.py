#!/usr/bin/env python3

from __future__ import division, print_function
import numpy as np
import os
import nibabel as nib
import json
import sys
import tarfile
from argparse import ArgumentParser
import JSON_templates


parser = ArgumentParser()
parser.add_argument("-i", "--input",
                    help="Execution workflow output file (predictions file)) to be validated", required=True)
parser.add_argument("-com", "--community_id",
                    help="OEB community id or community label of benchmarking community", required=True)
parser.add_argument("-c", "--challenges_ids", nargs='+',
                    help="Challenge(s) name(s) selected by the user", required=True)
parser.add_argument("-p", "--participant_id",
                    help="OEB id or name of the tool/model used for prediction", required=True)
parser.add_argument("-e", "--event_id",
                    help="OEB id or name of the benchmarking event", required=True)
## parser.add_argument("-pr", "--public_ref_dir",
##                     help="Directory containing public reference file(s)", required=False)
parser.add_argument("-g", "--goldstandard_dir",
                    help="Directory containing ground truth files", required=True)
#parser.add_argument("-t", "--template_path", help="Input path to the JSON template file with the minimal data for the aggregation step to obtain the minimal benchmark data 
args = parser.parse_args()

def extract_tarfile(input_file, extract_path):
    """
    Extracts a tar file to the specified path.
    """
    with tarfile.open(input_file, 'r:gz') as tar:
        tar.extractall(path=extract_path)
        return [os.path.join(extract_path, member.name) for member in tar.getmembers() if member.isfile()]

def main(args):
    # input parameters
    input_file = args.input
    community = args.community_id  # <- string 'EuCanImage'
    challenge = args.challenges_ids # in case of more than one challenge iterate below
    participant_id = args.participant_id  # <- string 'SynthSeg'
    event = args.event_id
    gt_path = args.goldstandard_dir # path to ground truth folder containing gt_*.nii.gz files
    # pr_path = args.public_ref_dir

    ### -------------------------------------------------------------------------------------------------------
    # UNTAR IMPUT FILE (single tar file of multiple nifti files)
    # Directory were the .gz files are going to be extracted
    untar_dir = os.path.join(os.path.dirname(input_file), 'participant_files')
        
    # Check if the directory already exists
    if not os.path.exists(untar_dir):
        # Create the directory
        os.makedirs(untar_dir)
        extracted_files = extract_tarfile(input_file, untar_dir)
    else:
        print(f"INFO: Directory {untar_dir} already exists.")
        extracted_files = [os.path.join(untar_dir, f) for f in os.listdir(untar_dir) if os.path.isfile(os.path.join(untar_dir, f))]
    
    print(f"INFO: Extracting files to {untar_dir}. Extracted files: {extracted_files}")
    ### -------------------------------------------------------------------------------------------------------


    ### validate_file(file, community, event, challenge, participant_name, outdir, gt_path, pr_path)
    # ...

    
    # Validate gzip and filenames of extracted participant file(s) for the event in a sorted way
    brain_files = sorted([f for f in os.listdir(untar_dir)
        if f.startswith('brain_') and f.endswith('.nii.gz')])

    if not brain_files:
        print("ERROR: No files matching the pattern 'brain_*.nii.gz' were found.")
        sys.exit(1)
    else:
        unmatch_files = [f for f in os.listdir(untar_dir) if not f.startswith('brain_') or not f.endswith('.nii.gz')]
        if unmatch_files:
            print(f"Found {len(unmatch_files)} file(s) that do not match the filename convention: {unmatch_files}")


    # Validate participant file(s) against ground truth files
    # get all nii.gz files in gt_path folder
    gt_files = sorted([os.path.join(gt_path, f) for f in os.listdir(gt_path) if f.startswith('gt_') and f.endswith('.nii.gz')])
    print(f"INFO: Found GT files: {gt_files}")
    
    
    for brain_file, gt_file in zip(extracted_files, gt_files):
        try:
            print(f"INFO: Loading brain file: {brain_file}")
            print(f"INFO: Loading GT file: {gt_file}")

            # Check the type of the variables
            brain_img = nib.load(brain_file)
            gt_img = nib.load(gt_file)

            if brain_img.header.get_zooms() != gt_img.header.get_zooms():
                print(f"ERROR: Spacing of {brain_file} does not match {gt_file}.")
                sys.exit(1)

            if brain_img.get_fdata().shape != gt_img.get_fdata().shape:
                print(f"ERROR: Dimensions of {brain_file} do not match {gt_file}.")
                sys.exit(1)

        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        except nib.filebasedimages.ImageFileError as e:
            print(f"ERROR: Issue loading NIfTI file. {e}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: Unexpected error: {e}")
            sys.exit(1)

    # Other validation tasks can be included

    # Once validated; emit the validation file
    # Create the output path filename using the input file's basename
    #output_filename = f"validated_{os.path.basename(input_file).replace('.gz', '')}.json"
    output_filename = f"validated_participant_data.json"

    data_id = f"{community}:{event}_{participant_id}" 
    validated = True
    output_json = JSON_templates.write_participant_dataset(
        data_id, community, challenge, participant_id, validated)

    # print validated input file
    with open(output_filename, 'w') as f:
        json.dump(output_json, f, sort_keys=True,
                  indent=4, separators=(',', ': '))

    # Only pass if all input files are valid
    if validated:
        sys.exit(0)
    else:
        sys.exit("ERROR: One or more of the submitted files don't comply with EuCanImage specified format!")


if __name__ == '__main__':

    main(args)
