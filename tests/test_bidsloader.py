import unittest
import os
import bids
from stroke.bids_loader import BIDSLoader

bids.config.set_option("extension_initial_dot", True)  # bids warning suppression


class TestBIDSLoader(unittest.TestCase):
    def test_BIDSLoader_init_empty(self):
        self.assertRaises(TypeError, BIDSLoader)
        return

    def test_BIDSLoader_init_empty_data(self):
        test_dir = os.path.dirname(__file__)
        root_dir = os.path.join(test_dir, "bids_sample/train")
        self.assertRaises(TypeError, BIDSLoader, root_dir=root_dir)
        return

    def test_BIDSLoader_init_empty_target(self):
        test_dir = os.path.dirname(__file__)
        root_dir = os.path.join(test_dir, "bids_sample/train")
        data_ent = [{"subject": "123"}]
        self.assertRaises(
            TypeError, BIDSLoader, root_dir=root_dir, data_entities=data_ent
        )
        return

    def test_BIDSLoader_init_correct(self):
        test_dir = os.path.dirname(__file__)
        root_dir = os.path.join(test_dir, "bids_sample/train")
        data_ent = [{"subject": "123"}]
        target_ent = [{"subject": "123"}]
        batch_size = 2

        bdc = BIDSLoader(
            root_dir=root_dir,
            data_entities=data_ent,
            target_entities=target_ent,
            batch_size=batch_size,
        )
        self.assertEqual(bdc.root_dir, root_dir)
        self.assertEqual(bdc.data_entities, data_ent)
        self.assertEqual(bdc.target_entities, target_ent)
        self.assertEqual(bdc.batch_size, batch_size)
        return

    def test_BIDSLoader_init_dataisderivatives_blank(self):
        test_dir = os.path.dirname(__file__)
        root_dir = os.path.join(test_dir, "bids_sample/train")
        data_ent = [
            {"subject": "123", "session": "5"},
            {"subject": "456", "session": "5"},
        ]
        target_ent = [{"subject": "123"}]
        batch_size = 2

        bdc = BIDSLoader(
            root_dir=root_dir,
            data_entities=data_ent,
            target_entities=target_ent,
            batch_size=batch_size,
        )
        self.assertEqual(len(bdc.data_derivatives_names), len(bdc.data_entities))
        self.assertEqual(len(bdc.data_derivatives_names), len(data_ent))
        return

    def test_BIDSLoader_init_dataisderivatives_input(self):
        test_dir = os.path.dirname(__file__)
        root_dir = os.path.join(test_dir, "bids_sample/train")
        data_ent = [
            {"subject": "123", "session": "5"},
            {"subject": "456", "session": "5"},
        ]
        target_ent = [{"subject": "123"}, {"subject": "429"}]

        bdc = BIDSLoader(
            root_dir=root_dir,
            data_entities=data_ent,
            target_entities=target_ent,
            data_derivatives_names=["test1", None],
        )
        self.assertTrue(bdc.data_is_derivatives[0])
        self.assertFalse(bdc.data_is_derivatives[1])
        return

    def test_BIDSLoader_init_targetisderivatives_blank(self):
        test_dir = os.path.dirname(__file__)
        root_dir = os.path.join(test_dir, "bids_sample/train")
        data_ent = [
            {"subject": "123", "session": "5"},
            {"subject": "456", "session": "5"},
        ]
        target_ent = [{"subject": "123"}, {"subject": "429"}]

        bdc = BIDSLoader(
            root_dir=root_dir,
            data_entities=data_ent,
            target_entities=target_ent,
            data_derivatives_names=["test1", None],
        )
        self.assertEqual(len(bdc.target_derivatives_names), len(target_ent))
        self.assertEqual(len(bdc.target_derivatives_names), len(bdc.target_entities))
        return

    def test_BIDSLoader_init_targetisderivatives_input(self):
        test_dir = os.path.dirname(__file__)
        root_dir = os.path.join(test_dir, "bids_sample/train")
        data_ent = [
            {"subject": "123", "session": "5"},
            {"subject": "456", "session": "5"},
        ]
        target_ent = [{"subject": "123"}, {"subject": "429"}]

        bdc = BIDSLoader(
            root_dir=root_dir,
            data_entities=data_ent,
            target_entities=target_ent,
            target_derivatives_names=[None, "test1"],
        )
        self.assertTrue(bdc.target_is_derivatives[1])
        self.assertFalse(bdc.target_is_derivatives[0])
        return

    def test_getemptyentities_emptydict(self):
        empty_ents = BIDSLoader._get_empty_entities({})
        self.assertEqual(len(empty_ents), 0)
        return

    def test_getemptyentities_dict_withoutempty(self):
        ents = {"one": "1", "two": "2"}
        empty_ents = BIDSLoader._get_empty_entities(ents)
        self.assertEqual(len(empty_ents), 0)
        return

    def test_getemptyentities_dict_withempty(self):
        ents = {"one": "1", "two": "2", "three": ""}
        empty_ents = BIDSLoader._get_empty_entities(ents)
        self.assertEqual(len(empty_ents), 1)
        self.assertEqual(empty_ents[0], "three")
        return

    def test_getfullentities_allempty(self):
        ents = {}
        full_ents = BIDSLoader._get_full_entities(ents)
        self.assertEqual(len(full_ents), 0)
        return

    def test_getfullentities_allfull(self):
        ents = {"one": "1", "two": "2", "five": "3"}
        full_ents = BIDSLoader._get_full_entities(ents)
        self.assertEqual(ents, full_ents)
        return

    def test_getfullentities_someempty(self):
        ents = {"one": "1", "two": "2", "five": "", "eight": None}
        expected_out = {"one": "1", "two": "2"}
        full_ents = BIDSLoader._get_full_entities(ents)
        self.assertEqual(full_ents, expected_out)
        return

    def test_getmatchingimages_empty(self):
        self.assertRaises(TypeError, BIDSLoader.get_matching_images)
        return

    def test_getmatchingimages_reqents(self):
        req_ents = {"subject": "001"}
        test_directory = os.path.dirname(__file__)
        bids_dataset = bids.BIDSLayout(
            root=os.path.join(test_directory, "bids_sample/train")
        )
        image_to_match = bids_dataset.get(extension="nii.gz")[0]
        returned_images = BIDSLoader.get_matching_images(
            image_to_match=image_to_match,
            bids_dataset=bids_dataset,
            required_entities=req_ents,
        )
        # Test directory has 4 images per subject and we shouldn't get the
        # input image back
        self.assertEqual(len(returned_images), 3)
        return

    def test_getmatchingimages_matching_ents(self):
        req_ents = {"subject": "001"}
        matching_ents = ["session"]
        test_directory = os.path.dirname(__file__)
        bids_dataset = bids.BIDSLayout(
            root=os.path.join(test_directory, "bids_sample/train")
        )
        image_to_match = bids_dataset.get(extension="nii.gz")[0]
        returned_session = BIDSLoader.get_matching_images(
            image_to_match=image_to_match,
            bids_dataset=bids_dataset,
            matching_entities=matching_ents,
            required_entities=req_ents,
        )
        # Test directory has 4 images split across two sessions & two modalities
        # Check that we get session correctly
        self.assertEqual(len(returned_session), 1)
        ses_im = returned_session[0]
        self.assertEqual(
            ses_im.get_entities()["session"], image_to_match.get_entities()["session"]
        )
        return

    def test_length(self):
        test_directory = os.path.dirname(__file__)
        root_dir = os.path.join(test_directory, "bids_sample/train")
        target_derivatives_names = ["test1"]
        bdc = BIDSLoader(
            root_dir=root_dir,
            data_entities=[
                {"suffix": "T1w", "session": "", "subject": ""},
                {"suffix": "FLAIR"},
            ],
            target_entities=[{"suffix": "FLAIR"}],
            target_derivatives_names=target_derivatives_names,
        )
        self.assertEqual(len(bdc), 4)
        return

    def test_load_sample(self):
        # matching_ents = ['subject', 'session']
        test_directory = os.path.dirname(__file__)
        root_dir = os.path.join(test_directory, "bids_sample/train")
        target_derivatives_names = ["test1"]
        bdc = BIDSLoader(
            root_dir=root_dir,
            data_entities=[
                {"suffix": "T1w", "session": "", "subject": ""},
                {"suffix": "FLAIR"},
            ],
            target_entities=[{"suffix": "FLAIR"}],
            target_derivatives_names=target_derivatives_names,
        )
        # Load data
        data, target = bdc.load_sample(0)
        self.assertEqual(data.shape, (2, 1, 1, 1))
        self.assertEqual(target.shape, (1, 1, 1, 1))
        return

    def test_load_batch(self):
        test_directory = os.path.dirname(__file__)
        root_dir = os.path.join(test_directory, "bids_sample/train")
        target_derivatives_names = ["test1"]
        bdc = BIDSLoader(
            root_dir=root_dir,
            data_entities=[
                {"suffix": "T1w", "session": "", "subject": ""},
                {"suffix": "FLAIR"},
            ],
            target_entities=[{"suffix": "FLAIR"}],
            target_derivatives_names=target_derivatives_names,
        )
        # Load data
        data, target = bdc.load_batch([0, 1])
        self.assertEqual(data.shape, (2, 2, 1, 1, 1))
        self.assertEqual(target.shape, (2, 1, 1, 1, 1))
        return

    def test_load_tuple(self):
        test_directory = os.path.dirname(__file__)
        root_dir = os.path.join(test_directory, "bids_sample/train")
        target_derivatives_names = ["test1"]
        bdc = BIDSLoader(
            root_dir=root_dir,
            data_entities=[
                {"suffix": "T1w", "session": "", "subject": ""},
                {"suffix": "FLAIR"},
            ],
            target_entities=[{"suffix": "FLAIR"}],
            target_derivatives_names=target_derivatives_names,
        )
        data = bdc.load_image_tuple(bdc.data_list[0])
        self.assertEqual(data.shape, (2, 1, 1, 1))
        self.assertEqual(data[0], 1)
        self.assertEqual(data[1], 1)
        target = bdc.load_image_tuple(bdc.target_list[0])
        self.assertEqual(target.shape, (1, 1, 1, 1))
        self.assertEqual(target[0], 1)
