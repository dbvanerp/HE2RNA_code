"""
HE2RNA: Match RNAseq data from TCGA with whole-slide images
Copyright (C) 2020  Owkin Inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from constant import PATH_TO_TILES, PATH_TO_TRANSCRIPTOME

PATH_TO_TILES = "/gpfs/work4/0/prjs1086/pepsi/data/processed/tcga_resnet_feats"
PATH_TO_TRANSCRIPTOME = "/gpfs/work4/0/prjs1086/pepsi/data/raw/tcga_rna/"

class TranscriptomeDataset:
    """A class for dealing with RNAseq data and matching them with available
    slides. Note: high memory usage when loading many transcriptomes (>1000).

    Args:
        projectname (list): If None, all TCGA projects are included.
        genes (list or None): list of genes Ensembl IDs. If None, all
            available genes are used.
    """

    def __init__(self, projectname=None, genes=None):

        self.projectname = projectname
        self.genes = genes

        # transcriptome_metadata = pd.read_csv(
        #     os.path.join(
        #         'metadata',
        #         'samples_description.csv'),
        #     sep='\t')
        transcriptome_metadata = pd.read_csv("/home/dvanerp/pepsi/data/raw/manifests/new_keyfiles/new_master_keyfile.csv", sep=',')
        
        # Instead of inferring Sample.ID and Case.ID from slide_filename like below, I requested the Case.ID and Sample.ID from the GDC API and added them to the metadata file.
   
        # transcriptome_metadata['Sample.ID'] = transcriptome_metadata['slide_filename'].apply(lambda x: x[:16])

        # # Set 'Case.ID' based on 'File.Name':
        # transcriptome_metadata['Case.ID'] = transcriptome_metadata['slide_filename'].apply(lambda x: x[:12])

        # # For now, this sets all as 'Primary Tumor':
        # transcriptome_metadata['Sample.Type'] = 'Primary Tumor'
        
        # Select primary tumor samples from the chosen project
        if self.projectname is not None:
            directories = [
                project for project in self.projectname]
            self.transcriptome_metadata = transcriptome_metadata.loc[
                (transcriptome_metadata['Project.ID'].isin(directories))  &
                (transcriptome_metadata['Sample.Type'] == 'Primary Tumor')].copy()
        else:
            self.transcriptome_metadata = transcriptome_metadata.loc[
                transcriptome_metadata['Sample.Type'] == 'Primary Tumor'].copy()

        self.image_metadata = self._get_infos_on_tiles(self.projectname)
        self._match_data()

    @classmethod
    def from_saved_file(cls, path, projectname=None, genes=None):
        """Build TranscriptomeDataset instance from a saved csv file.
        """
        if genes is None:
            usecols = None
        else:
            usecols = list(genes) + ['File.ID', 'Sample.ID', 'Case.ID', 'Project.ID']
        transcriptomes = pd.read_csv(path, usecols=usecols)
        if projectname is None:
            projectname = transcriptomes['Project.ID']
        else:
            transcriptomes = transcriptomes.loc[transcriptomes['Project.ID'].isin(projectname)]
        genes = [col for col in transcriptomes.columns if col.startswith('ENSG')]
        dataset = cls(projectname, genes)
        transcriptomes.sort_values('Sample.ID', inplace=True)
        transcriptomes.reset_index(inplace=True, drop=True)
        dataset.transcriptomes = transcriptomes
        return dataset

    def _get_infos_on_tiles(self, subdirs, zoom='0.50_mpp'):
        """Find all slides tiled at a given level of a TCGA project and return a
        dataframe with their metadata.
        """

        if subdirs is not None:
            df = []
            for subdir in tqdm(subdirs, desc="getting metadata per slide feature", mininterval=4, total=len(subdirs)):
                dir_tiles = os.path.join(PATH_TO_TILES, subdir, zoom)
                filenames = [f for f in os.listdir(dir_tiles) if f.endswith('.npy') and 'mask' not in f]
                case_ids = [f[:12] for f in filenames]
                sample_ids = [f[:16] for f in filenames]
                full_ids = [f.split('.')[0] for f in filenames]

                df.append(pd.DataFrame(
                    {'Project.ID': subdir, 'Case.ID': case_ids, 'Sample.ID_image': sample_ids,
                     'ID': full_ids, 'Slide.ID': filenames}))
            return pd.concat(df)
        else:
            print("No projectname filters provided, getting subdirs for all TCGA projects in PATH_TO_TILES")
            subdirs = []
            for subdir in os.listdir(PATH_TO_TILES):
                if os.path.isdir(os.path.join(PATH_TO_TILES, subdir)) and subdir.startswith('TCGA'):
                    subdirs.append(subdir)
            return self._get_infos_on_tiles(subdirs)

    def _match_data(self):
        """Associate transcriptomes with availables slides.
        """
        self.transcriptome_metadata['Sample'] = self.transcriptome_metadata['Sample.ID'].apply(
            lambda x: x[:-1])
        self.image_metadata['Sample'] = self.image_metadata['Sample.ID_image'].apply(
            lambda x: x[:-1])
        self.transcriptome_metadata.drop('Project.ID', axis=1, inplace=True)
        self.metadata = self.transcriptome_metadata.merge(
            self.image_metadata[['Project.ID', 'Sample', 'Sample.ID_image', 'ID', 'Slide.ID']],
            on='Sample')
        # If several transcriptomes can be associated with a slide, pick only one.
        self.metadata = self.metadata.groupby('Slide.ID').first().reset_index()
        self.metadata.sort_values('Sample.ID', inplace=True)
        self.metadata.reset_index(inplace=True, drop=True)

    def load_transcriptomes(self, csv_path=None):
        """Select transcriptomic data of the selected project and genes.
        """
        if csv_path is None:
            csv_path = Path(PATH_TO_TRANSCRIPTOME) / 'transcriptome_fpkmuq_allsamps.csv'
        else:
            csv_path = Path(csv_path)

        df = pd.read_csv(
            csv_path,
            sep='\t',
            usecols=self.genes,
            index_col=0)

        df['File.ID'] = df.index
        df = df.merge(self.metadata[['File.ID', 'Sample.ID',
                                     'Case.ID', 'Project.ID']],
                      on='File.ID', how='inner')
        df.sort_values('Sample.ID', inplace=True)
        df.reset_index(inplace=True, drop=True)
        self.transcriptomes = df

        
def main():
    parser = argparse.ArgumentParser(
        description="Build or reuse aggregated TCGA transcriptome data and load it into a TranscriptomeDataset.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(PATH_TO_TRANSCRIPTOME) / 'transcriptome_fpkmuq_allsamps.csv',
        help="Path to an existing aggregated transcriptome file. Defaults to PATH_TO_TRANSCRIPTOME/transcriptome_fpkmuq_allsamps.csv.")
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force regeneration of the aggregated transcriptome file even if it already exists.")
    args = parser.parse_args()

    target_csv = args.input_csv.expanduser().resolve()
    source_dir = Path(PATH_TO_TRANSCRIPTOME)
    print("Final csv path:", source_dir / 'all_transcriptomes_fpkm_uq.csv')
    if args.force_rebuild or not target_csv.exists():
        print(f"Generating aggregated transcriptome file at {target_csv}")
        target_csv.parent.mkdir(parents=True, exist_ok=True)
        df = []
        tsv_files = list(source_dir.rglob('**/*.tsv'))
        for f in tqdm(tsv_files, desc="Loading transcriptomes from folders", total=len(tsv_files), mininterval=3):
            df_ = pd.read_csv(
                f,
                sep='\t',
                comment='#',                     # skip the first `# gene-model…` line
                usecols=['gene_id', 'fpkm_uq_unstranded']
            )
            df_ = df_.loc[df_['gene_id'].str.startswith('ENSG')].set_index('gene_id')
            df_.columns = [f.parent.name]        # use the File ID folder as the sample name
            df.append(df_.T)
        print("Concatenating transcriptomes")
        df = pd.concat(df)
        print("Concatenated transcriptomes")
        print(df.head())
        print("Shape of df:", df.shape)
        df.to_csv(target_csv, index=True, sep='\t')
        print("Saved transcriptomes to csv")
    else:
        print(f"Using existing aggregated transcriptome file at {target_csv}")

    print("Initializing TranscriptomeDataset")
    
    dataset = TranscriptomeDataset()
    print("Done, now loading transcriptomes into dataset")
    print(f"Shape of initialized TranscriptomeDataset: {dataset.transcriptomes.shape if hasattr(dataset, 'transcriptomes') and dataset.transcriptomes is not None else 'Not loaded yet'}")
    print(f"Columns in initialized TranscriptomeDataset: {dataset.transcriptomes.columns if hasattr(dataset, 'transcriptomes') and dataset.transcriptomes is not None else 'Not loaded yet'}")
    
    dataset.load_transcriptomes(csv_path=target_csv)
    print("Loaded transcriptomes into dataset")
    print(f"Shape of loaded transcriptomes: {dataset.transcriptomes.shape if hasattr(dataset, 'transcriptomes') and dataset.transcriptomes is not None else 'Not loaded yet'}")
    # print(f"Columns of loaded transcriptomes: {dataset.transcriptomes.columns if hasattr(dataset, 'transcriptomes') and dataset.transcriptomes is not None else 'Not loaded yet'}")
    
    dataset.transcriptomes.to_csv(source_dir / 'all_transcriptomes_fpkm_uq.csv', index=False)
    print("Saved all_transcriptomes to csv")
    print("Done")

if __name__ == '__main__':

    main()