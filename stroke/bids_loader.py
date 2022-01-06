import warnings
from collections import defaultdict
import numpy as np
import bids

bids.config.set_option("extension_initial_dot", True)  # bids warning suppression
from bids import BIDSLayout
from bids.layout.models import BIDSImageFile


class BIDSLoader:
    """
    BIDS-compatible data loader used for classifying BIDS datasets.
    Parameters
    ----------
    root_dir : str
        BIDS root directory; subject directories should be immediately below this (e.g. root_dir/sub-123)
    data_entities : list [dict]
        List of dictionaries, where each dictionary contains BIDS entities that will uniquely match data. Multiple
        dictionaries should be used to if multiple files will be used for prediction.
        Empty entries ({'subject': ''}) indicate that that entry should match across samples (e.g., [{'session': ''}]
        would ensure that entities from a returned sample are from the same session, but that any value is valid.
        For example: ({'subject': '', 'session': '1', 'desc': 'Normalized'}, {'session': '1', 'desc': 'defaced'}) would
        return samples of two images: The first could be 'sub-123_ses-1_desc-Normalized.nii.gz', and the second would be
        'sub-123_ses-1_desc-defaced.nii.gz'. The subject entity matches, session is restricted to "1", and the
        description entitiy is used to differentiate between them.
    target_entities : list [dict]
        Same as data_entities, but for the prediction target.
    batch_size : int
        Optional. Size of the batch to train the estimator. Default: 1.
    data_derivatives_names : list [str]
        Optional. If an entry in data_entities is BIDS derivatives data, its name should be listed here. Entries
        that don't correspond to derivatives should be listed as None. Default: [None for _ in data_entities]
    target_derivatives_names : list [str]
        Optional. If an entry in target_entities is BIDS derivatives data, its name should be listed here. Entries
        that don't correspond to derivatives should be listed as None. Default: [None for _ in target_entities]
    root_list : list
        Reserved. Not yet implemented. List of BIDS root directories, if data must be loaded from different BIDS
        directories. There must be exactly len(data_entities) + len(target_entities) entries in the list, with the
        order corresponding to the order of the data_entities, followed by the target_entities.
    label_names : list [str]
        Names of the values of the target, if any.
    """

    def __init__(
        self,
        root_dir: str,
        data_entities: list,
        target_entities: list,
        batch_size: int = 1,
        data_derivatives_names: list = None,
        target_derivatives_names: list = None,
        root_list: list = None,
        label_names: list = None,
    ):

        self.root_dir = root_dir

        if isinstance(data_entities, list):
            self.data_entities = data_entities
        elif isinstance(data_entities, dict) or isinstance(data_entities, defaultdict):
            self.data_entities = [data_entities]
        else:
            raise (TypeError("data_entities should be a list of dicts"))

        if isinstance(target_entities, list):
            self.target_entities = target_entities
        elif isinstance(target_entities, dict) or isinstance(
            target_entities, defaultdict
        ):
            self.target_entities = [target_entities]
        else:
            raise (TypeError("target_entities should be a list of dicts"))

        self.batch_size = batch_size

        # Deal with data + derivatives
        if data_derivatives_names is None:
            self.data_derivatives_names = [None for _ in self.data_entities]
        else:
            self.data_derivatives_names = data_derivatives_names
        self.data_bids = []
        default_data = bids.BIDSLayout(root=root_dir)
        for name in self.data_derivatives_names:
            if name is None:
                self.data_bids.append(default_data)
            else:
                self.data_bids.append(
                    bids.BIDSLayout(root=root_dir, derivatives=True).derivatives[name]
                )
        self.data_is_derivatives = [s is not None for s in self.data_derivatives_names]

        # Deal with target + derivatives
        if target_derivatives_names is None:
            self.target_derivatives_names = [
                None for _ in self.target_entities
            ]  # change to list
        else:
            self.target_derivatives_names = target_derivatives_names
        self.target_bids = []
        for name in self.target_derivatives_names:
            if name is None:
                self.target_bids.append(default_data)
            else:
                self.target_bids.append(
                    bids.BIDSLayout(root=root_dir, derivatives=True).derivatives[name]
                )
        self.target_is_derivatives = [
            s is not None for s in self.target_derivatives_names
        ]

        self.unmatched_image_list = []
        self.unmatched_target_list = []
        self._loader_prep()
        if len(self.data_list) > 0:
            self.data_shape = self.data_list[0][0].get_image().shape
        if len(self.target_list) > 0:
            self.target_shape = self.target_list[0][0].get_image().shape

        if root_list is not None:
            raise (
                NotImplementedError(
                    "Processing root list has not yet been implemented."
                )
            )

        self.label_names = label_names
        self._prediction_label_names = self.label_names  # RAMP convention
        return

    def _loader_prep(self):
        """
        Prepares the loader to satisfy the required + matching entities.

        Returns
        -------
        None
        """

        # First get empty entities; these are used to flag that entities must
        # match
        entities_to_match = set()  # Use set to avoid having to check for uniqueness
        data_entities_full = []
        target_entities_full = []

        for ents in self.data_entities:
            empty_ents = self._get_empty_entities(ents)
            entities_to_match.update(empty_ents)
            data_entities_full.append(self._get_full_entities(ents))

        for ents in self.target_entities:
            empty_ents = self._get_empty_entities(ents)
            entities_to_match.update(empty_ents)
            target_entities_full.append(self._get_full_entities(ents))
        self.entities_to_match = entities_to_match

        # Create file list
        self.data_list = []
        self.target_list = []
        bids_set = bids.BIDSLayout(
            root=self.root_dir, derivatives=self.data_is_derivatives[0]
        )
        if self.data_is_derivatives[0]:
            bids_set = bids_set.derivatives[self.data_derivatives_names[0]]

        bids_data = bids_set.get(**data_entities_full[0])

        # For each image returned, get images for other sets in data_entitites
        for im in bids_data:
            sample_list = [im]
            for idx, data_entities in enumerate(data_entities_full[1:]):
                new_sample = self.get_matching_images(
                    image_to_match=im,
                    # +1 since we skip first
                    bids_dataset=self.data_bids[idx + 1],
                    matching_entities=entities_to_match,
                    required_entities=data_entities,
                )
                if len(new_sample) > 1:
                    warnings.warn(
                        f"Image matching returned more than one match for data; make either required_entities or "
                        f"matching_entities to be more specific. {im}"
                    )
                elif len(new_sample) == 0:
                    self.unmatched_image_list.append(im)
                    warnings.warn(f"No match found for image {im}")
                else:
                    sample_list.append(new_sample[0])
            self.data_list.append(tuple(sample_list))

            # Similarly for target data
            sample_list = []
            for idx, target_entities in enumerate(target_entities_full):
                new_sample = self.get_matching_images(
                    image_to_match=im,
                    bids_dataset=self.target_bids[idx],
                    matching_entities=entities_to_match,
                    required_entities=target_entities,
                )
                if len(new_sample) > 1:
                    warnings.warn(
                        "Image matching returned more than one match for target; either make "
                        f"required_entities or matching_entities more specific. (image: {im})"
                    )
                elif len(new_sample) == 0:
                    self.unmatched_target_list.append(im)
                    warnings.warn(f"No match found for image {im}")
                else:
                    sample_list.append(new_sample[0])
            self.target_list.append(tuple(sample_list))

        if len(self.unmatched_image_list):
            warnings.warn("Not all images had matches.")
        return

    @staticmethod
    def get_matching_images(
        image_to_match: BIDSImageFile,
        bids_dataset: BIDSLayout,
        matching_entities: list = None,
        required_entities: dict = None,
    ):
        """
        Returns a list of images from the BIDS dataset that has the specified required_entities and has the same
        value for entities listed in matching_entities as the image_to_match.
        Example: for an image "sub-123_ses-1_T1w.nii" with matching_entities ['ses'] and required_entities
        {'suffix': 'FLAIR'}, the image "sub-123_ses-1_FLAIR.nii" would match, but "sub-123_ses-2_FLAIR.nii" would not.
        Parameters
        ----------
        required_entities: dict
            Entity-value dictionary that are required.
        matching_entities: list
            List of entities that must match, if present, between the previous image and the one to fetch.
        image_to_match: BIDSImageFile
            Image to use as reference for matching_entities.
        bids_dataset: BIDSLayout
            BIDS dataset from which to fetch the new image.

        Returns
        -------
        list [BIDSImageFile]
            BIDS image file matching the input specifications. Empty if there are no matches.
        """

        if matching_entities is None:
            matching_entities = []
        if required_entities is None:
            required_entities = {}

        ents_to_match = {}
        im_entities = image_to_match.get_entities()
        for k in matching_entities:
            if k in im_entities.keys():
                ents_to_match[k] = im_entities[k]
        potential_matches = bids_dataset.get(**required_entities, **ents_to_match)
        # Go through each potential image; remove those that don't match
        potential_idx = []
        for idx, potential_im in enumerate(potential_matches):
            potential_im_ents = potential_im.get_entities()
            for entity, value in ents_to_match.items():
                if (
                    entity not in potential_im_ents.keys()
                    or value != potential_im_ents[entity]
                ):
                    continue
            else:
                if potential_im != image_to_match:
                    potential_idx.append(idx)
        return [potential_matches[i] for i in potential_idx]

    @staticmethod
    def _get_empty_entities(ents: dict):
        """
        Returns list of empty entities. Empty entities are those that are either None or zero-length strings.
        Complements _get_full_entities
        Parameters
        ----------
        ents : dict
            Dictionary to check.

        Returns
        -------
        list
            List of keywords that were found to have empty values.
        """
        empty_ents = []
        for k, v in ents.items():
            if len(v) == 0:
                empty_ents.append(k)
        return empty_ents

    @staticmethod
    def _get_full_entities(ents: dict):
        """
        Returns dictionary without empty entities. Empty entities are those that are either None or zero-length strings.
        Complements _get_empty_entities
        Parameters
        ----------
        ents : dict
            Dictionary to check

        Returns
        -------
        dict
            Dictionary without empty entries
        """
        full_dict = {}
        for k, v in ents.items():
            if v is not None and len(v) != 0:
                full_dict[k] = v
        return full_dict

    @staticmethod
    def load_image_tuple(image_tuple: tuple, dtype=np.float32):
        """
        Loads the tuple and returns it in an array
        Parameters
        ----------
        image_tuple : tuple (BIDSImageFile,)
            Tuple of BIDSImageFile to be loaded and returned in an array
        Returns
        -------
        np.array
            Loaded data
        """
        data_shape = image_tuple[0].get_image().shape
        if dtype is not bool:
            data = np.zeros((len(image_tuple), *data_shape), dtype=dtype)
            for idx, im in enumerate(image_tuple):
                data[idx, ...] = np.array(im.get_image().get_fdata(), dtype=dtype)
        else:
            num_bytes = int(np.ceil(np.prod(data_shape) / 8))
            data = np.zeros((len(image_tuple), num_bytes), dtype=np.uint8)
            for idx, im in enumerate(image_tuple):
                tmp_dat = np.array(im.get_image().get_fdata(), dtype=dtype)
                data[idx, ...] = np.packbits(tmp_dat)
        return data

    @staticmethod
    def load_image_tuple_list(image_list: list, dtype=np.float32):
        """
        Loads each image in the tuple and returns in a single array; different tuples in the list are assumed to be
        batches. The returned array will be of shape (len(image_list), len(image_tuple), *image.shape
        Parameters
        ----------
        image_list : list [tuple]
            List of tuples containing BIDSImageFile

        Returns
        -------
        np.array
            Loaded data.
        """
        num_batch = len(image_list)
        num_dim = len(image_list[0])
        data_shape = image_list[0][0].get_image().shape
        if dtype is not bool:
            data = np.zeros((num_batch, num_dim, *data_shape), dtype=dtype)
            for idx, image_tuple in enumerate(image_list):
                data[idx, ...] = BIDSLoader.load_image_tuple(image_tuple, dtype=dtype)
        else:
            num_bytes = int(np.ceil(np.prod(data_shape) / 8))
            data = np.zeros((num_batch, num_dim, num_bytes), dtype=np.uint8)
            for idx, image_tuple in enumerate(image_list):
                data[idx, ...] = BIDSLoader.load_image_tuple(image_tuple, dtype=dtype)
        return data

    def load_sample(self, idx: int):
        """
        Loads the sample at idx.
        Parameters
        ----------
        idx : int
            Index of the sample to load. Max valid value is len(BIDSClassifier)-1

        Returns
        -------
        np.array
            Array of shape (num_data, *image.shape) containing the data.
        np.array
            Array of shape (num_target, *image.shape) containing the target.
        """
        data = np.zeros((len(self.data_entities), *self.data_shape), dtype=np.float32)
        target = np.zeros(
            (len(self.target_entities), *self.target_shape), dtype=np.float32
        )

        for point_idx, point in enumerate(self.data_list[idx]):
            data[point_idx, ...] = point.get_image().get_fdata()
        for point_idx, point in enumerate(self.target_list[idx]):
            target[point_idx, ...] = point.get_image().get_fdata()
        return data, target

    def __len__(self):
        return len(self.data_list)

    def load_batch(self, indices: list):
        """
        Loads a batch of N images and returns the data/target in arrays.
        Parameters
        ----------
        indices : list [int]
            List of indices to load.
        Returns
        -------
        np.array
            Array of shape (len(indices), num_data, *image.shape) containing data.
        np.array
            Array of shape (len(indices), num_target, *image.shape) containing data.
        """
        data = np.zeros(
            (len(indices), len(self.data_entities), *self.data_shape), dtype=np.float32
        )
        target = np.zeros(
            (len(indices), len(self.target_entities), *self.target_shape),
            dtype=np.float32,
        )

        for i, idx in enumerate(indices):
            data[i, ...], target[i, ...] = self.load_sample(idx)
        return data, target
