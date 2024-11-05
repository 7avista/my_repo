#!/usr/bin/env python3

from __future__ import division
import io
import os
import json
import pandas as pd
import numpy as np
import tarfile
import nibabel as nib
from argparse import ArgumentParser
import JSON_templates
import segmentationmetrics as sm


def extract_tarfile(input_file, extract_path):
    """
    Extracts a tar file to the specified path.

    :param input_file: Path to the tar file
    :param extract_path: Directory to extract files into
    :return: List of extracted file paths
    """
    with tarfile.open(input_file, 'r:gz') as tar:
        tar.extractall(path=extract_path)
        return [os.path.join(extract_path, member.name) for member in tar.getmembers() if member.isfile()]
    
def main(args):

    # input parameters
    input_file = args.input
    goldstandard_dir = args.goldstandard_dir
    challenges_ids = args.challenges_ids
    event = args.event_id
    participant_id = args.participant_id
    community = args.community_id
    #output = args.output
    outdir = args.outdir

    print(f"INFO: participant input file {input_file}")
    print(f"INFO: Selected challenge(s) {challenges_ids}")

    # In case of more than one challenge
    # challenge = [c for c in challenges_ids if c.split('.')[0] == str(input_file).split('.')[1]]


    ### -------------------------------------------------------------------------------------------------------
    # Untar input file (single tar file of multiple nifti files)
    # Directory were the .gz files are going to be extracted
    untar_dir = os.path.join(os.path.dirname(input_file), 'participant_files')
    print(f"INFO: Untar dir {untar_dir}.")
        
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


    # Assuring the output path does exist
    if not os.path.exists(os.path.dirname(outdir)):
        try:
            os.makedirs(os.path.dirname(outdir))
            with open(outdir, mode="a"):
                pass
        except OSError as exc:
            print("OS error: {0}".format(exc) +
                  "\nCould not create output path: " + outdir)

    compute_metrics(input_file, goldstandard_dir,
                    challenges_ids, participant_id, community, event, outdir)


def compute_metrics(untar_dir, goldstandard_dir, challenge, participant_id,
                    community, event, outdir):

    input_file = args.input
    untar_dir = os.path.join(os.path.dirname(input_file), 'participant_files')
    gt_path = args.goldstandard_dir

    # Get and sort brain files
    brain_files = sorted([f for f in os.listdir(untar_dir)
                          if f.startswith('brain_') and f.endswith('.nii.gz')])
    brain_files = [os.path.join(untar_dir, f) for f in brain_files]
    print(f"Sorted brain files: {brain_files}")
    
    gt_files = sorted([f for f in os.listdir(gt_path)
                      if f.startswith('gt_') and f.endswith('.nii.gz')])

    print(f"INFO: Found GT files: {gt_files}")

    gt_files = [os.path.join(gt_path, f) for f in gt_files]
    print(f"Sorted brain files: {gt_files}")


    assert len(brain_files) == len(
        gt_files), f"ERROR: Number of submitted segmentation files {os.listdir(brain_files)} does not match number of ground truth files {len(gt_files)}!"

   
    # define array that will hold the full set of assessment datasets
    all_assessments = []
    # define list with keywords for return type of dataframe from match_with_gt()
    all_return_df_types = ["all_GT", "union", "intersection"]

    # ID prefix for assessment objects
    if isinstance(challenge, list):
        challenge = challenge[0]
    base_id = f"{community}:{event}_{challenge}_{participant_id}:"
    # Dict to store metric names and corresponding variables + stderr (which is currently not computed and set to 0)
    metrics_summary = {}


    # TODO implement:
    # other metrics
    all_samples = []
    for i, (brain, gt) in enumerate(zip(brain_files, gt_files)):
        brain_img = nib.load(brain)
        gt_img = nib.load(gt)
        zoom = gt_img.header.get_zooms()
        # get labels
        labels = np.unique(brain_img.get_fdata())[:10]
        dsc_patient = []
        for label in labels:  # TODO exclude background
            # get segmentation mask for current label
            brain_mask = brain_img.get_fdata() == label
            # get ground truth mask for current label
            gt_mask = gt_img.get_fdata() == label
            # https://pypi.org/project/segmentationmetrics/
            metrics = sm.SegmentationMetrics(
                brain_mask, gt_mask, zoom)  # compute metrics
            dsc_patient.append(metrics.dice)
        all_samples.append(dsc_patient)
    all_samples = np.array(all_samples)
    # compute mean and std
    per_label_mean = np.mean(all_samples, axis=0)
    per_label_std = np.std(all_samples, axis=0)
    for i, label in enumerate(labels):  # TODO exclude background
        metrics_summary[f"dsc_{label}"] = [
            per_label_mean[i], per_label_std[i]]

    # for the challenge, create all assessment json objects and append them to all_assessments
    for key, value in metrics_summary.items():
        object_id = base_id + key
        assessment_object = JSON_templates.write_assessment_dataset(
            object_id, community, challenge, participant_id, key, value[0], value[1])
        all_assessments.append(assessment_object)

    # once all assessments have been added, print to json file
    with io.open(outdir,
                 mode='w', encoding="utf-8") as f:
        jdata = json.dumps(all_assessments, sort_keys=True,
                           indent=4, separators=(',', ': '))
        f.write(jdata)
  


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-i", "--input",
                        help="List of execution workflow output files", required=True)
    parser.add_argument("-c", "--challenges_ids", nargs='+',
                        help="List of challenges ids selected by the user, separated by spaces", required=True)
    parser.add_argument("-g", "--goldstandard_dir",
                        help="dir that contains gold standard datasets for current challenge", required=True)
    parser.add_argument("-p", "--participant_id",
                        help="name of the tool used for prediction", required=True)
    parser.add_argument("-com", "--community_id",
                        help="name/id of benchmarking community", required=True)
    parser.add_argument("-e", "--event_id",
                        help="name/id of benchmarking event", required=True)
    parser.add_argument("-o", "--outdir", 
                        help="output path where assessment JSON files will be written", required=True)

    args = parser.parse_args()

    main(args)
