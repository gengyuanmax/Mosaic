"""Check the transport/state boundary against direct, unmodified Toolbox calls."""
import hashlib
import inspect
import io
import json
import platform
import unittest
from pathlib import Path

from server import GymService, ServiceError, DEFAULT_SITE, DEFAULT_SOURCE

CASES = [
    ('crop', {'x1': .2, 'y1': .15, 'x2': .81, 'y2': .9}, [0]),
    ('rotate', {'angle': 30}, [0]),
    ('resize', {'width': 273, 'height': 301}, [0]),
    ('flip', {'mode': 'vertical'}, [0]),
    ('apply_affine_transformation', {'rotation_angle': 20, 'scale': .85, 'translation_dx': 60, 'translation_dy': -30}, [0]),
    ('apply_homography_transformation', {'matrix': [1, .15, 0, .05, 1, 0, .0004, .00015, 1]}, [0]),
    ('draw_normalized_coordinate_grid', {'grid_rows': 4, 'grid_cols': 3}, [0]),
    ('compare_per_pixel_image_difference', {}, [0, 1]),
    ('make_collage', {'rows': 2, 'cols': 2}, [1, 0]),
    ('overlay_images', {'x': .22, 'y': .18}, [0, 1]),
]
REPORT = {'comparisons': []}


def pixels(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


class ToolboxBridgeTests(unittest.TestCase):
    def setUp(self):
        self.service = GymService()
        self.workspace = self.service.create(self.service.examples[0]['id'])
        self.sid = self.workspace['session_id']
        self.session = self.service.session(self.sid)
        self.direct = self.service.Toolbox()
        self.direct.reset(list(self.session.box.images_by_id().values()))

    def call(self, name, args, indices, box=None):
        ids = (box or self.direct).image_ids()
        return {'name': name, 'arguments': args, 'target_images': [ids[i] for i in indices]}

    def compare(self, call, label):
        before = {k: pixels(v) for k, v in self.session.box.images_by_id().items()}
        revision = self.session.revision
        response = self.service.preview(self.sid, call, revision)
        expected = self.direct.run([self.service.ToolCall(**call)])[0]
        self.assertEqual(response['error'], expected.error)
        self.assertIsNone(expected.error)
        self.assertEqual(response['new_image_ids'], expected.new_image_ids)
        self.assertEqual(response['tool_feedback'], expected.tool_feedback)
        actual = self.service.Image.open(io.BytesIO(self.service.png(self.sid, 'previews', response['preview_id'], 'rgba.png')))
        alpha = self.service.Image.open(io.BytesIO(self.service.png(self.sid, 'previews', response['preview_id'], 'alpha.png')))
        wanted = expected.produced_images[0]
        self.assertEqual(actual.mode, 'RGBA')
        self.assertEqual(actual.size, wanted.size)
        self.assertEqual(pixels(actual), pixels(wanted))
        self.assertEqual(pixels(alpha), pixels(wanted.getchannel('A')))
        self.assertEqual(before, {k: pixels(v) for k, v in self.session.box.images_by_id().items()})
        self.assertEqual(revision, self.session.revision)
        REPORT['comparisons'].append({'case': label, 'size': actual.size, 'rgba_sha256': pixels(actual), 'alpha_sha256': pixels(alpha), 'equal': True})
        return response

    def test_all_ten_tools_on_sources_and_transparent_results(self):
        for transparent in (False, True):
            for name, args, indices in CASES:
                with self.subTest(tool=name, transparent=transparent):
                    self.setUp()
                    if transparent:
                        # Real rotate output, including alpha=0 corners, becomes
                        # both inputs. No synthetic replacement of tool logic.
                        draft = self.service.preview(self.sid, self.call('rotate', {'angle': 27}, [0]), 0)
                        self.service.commit(self.sid, draft['preview_id'], 0)
                        image = self.session.box.images_by_id()[draft['image']['id']]
                        self.session.box.reset([image, image.copy()])
                        self.direct.reset([image, image.copy()])
                    self.compare(self.call(name, args, indices), name + ('/rgba' if transparent else '/source'))

    def test_chain_commits_exact_preview_pixels(self):
        for name, args, indices in [CASES[1], ('overlay_images', {'x': .22, 'y': .22}, [0, 2]), CASES[6]]:
            response = self.compare(self.call(name, args, indices), 'chain/' + name)
            self.service.commit(self.sid, response['preview_id'], self.session.revision)
            actual = self.session.box.images_by_id()[response['image']['id']]
            expected = self.direct.images_by_id()[response['image']['id']]
            self.assertEqual(pixels(actual), pixels(expected))

    def test_tool_errors_are_unchanged_and_do_not_mutate_workspace(self):
        invalid = [
            self.call('apply_homography_transformation', {'matrix': [0] * 9}, [0]),
            self.call('crop', {'x1': .9, 'x2': .1, 'y1': 0, 'y2': 1}, [0]),
            self.call('rotate', {'angle': 1000}, [0]),
            self.call('make_collage', {'rows': 1, 'cols': 1}, [0]),
        ]
        resize = self.service.preview(self.sid, self.call('resize', {'width': 111, 'height': 92}, [0]), 0)
        self.service.commit(self.sid, resize['preview_id'], 0)
        self.direct.run([self.service.ToolCall(**self.call('resize', {'width': 111, 'height': 92}, [0]))])
        invalid.append(self.call('compare_per_pixel_image_difference', {}, [0, 2]))
        for call in invalid:
            expected = self.direct.run([self.service.ToolCall(**call)])[0]
            response = self.service.preview(self.sid, call, self.session.revision)
            self.assertIsNotNone(expected.error)
            self.assertEqual(response['error'], expected.error)
            self.assertEqual(len(self.session.box.image_ids()), 3)

    def test_stale_preview_and_revision_cannot_be_committed(self):
        first = self.service.preview(self.sid, self.call('rotate', {'angle': 30}, [0]), 0)
        latest = self.service.preview(self.sid, self.call('rotate', {'angle': 40}, [0]), 0)
        with self.assertRaises(ServiceError) as caught:
            self.service.commit(self.sid, first['preview_id'], 0)
        self.assertEqual(caught.exception.status, 409)
        self.service.commit(self.sid, latest['preview_id'], 0)
        with self.assertRaises(ServiceError):
            self.service.preview(self.sid, self.call('rotate', {'angle': 20}, [0]), 0)
        before = {k: pixels(v) for k, v in self.direct.images_by_id().items()}
        self.service.undo(self.sid, 1)
        self.assertEqual(before, {k: pixels(v) for k, v in self.session.box.images_by_id().items()})

    def test_all_example_files_load_without_resizing_or_mutation(self):
        for example in self.service.examples:
            workspace = self.service.create(example['id'])
            session = self.service.session(workspace['session_id'])
            for item, image in zip(example['images'], session.box.images_by_id().values()):
                path = self.service.site / item['file']
                before = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.service.Image.open(path) as original:
                    self.assertEqual(image.size, original.size)
                    self.assertEqual(pixels(image), pixels(original.convert('RGBA')))
                self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())
            self.service.close(workspace['session_id'])

    def test_registry_and_import_are_actual_release_package(self):
        self.assertEqual(self.service.config()['tools'], self.direct.registry.list_tools())
        self.assertEqual({c[0] for c in CASES}, {t['name'] for t in self.service.specs})
        self.assertTrue(Path(inspect.getfile(self.service.Toolbox)).resolve().is_relative_to(DEFAULT_SOURCE.resolve()))

    def test_resource_rejection_does_not_resize_or_commit(self):
        draft = self.service.preview(self.sid, self.call('resize', {'width': 1024, 'height': 1024}, [0]), 0)
        self.service.commit(self.sid, draft['preview_id'], 0)
        with self.assertRaisesRegex(ServiceError, 'memory budget'):
            self.service.preview(self.sid, {'name': 'make_collage', 'arguments': {'rows': 8, 'cols': 8}, 'target_images': ['img_0003', 'img_0003']}, 1)
        self.assertEqual(self.session.box.images_by_id()['img_0003'].size, (1024, 1024))
        self.assertEqual(len(self.session.box.image_ids()), 3)


if __name__ == '__main__':
    import cv2
    import numpy
    import PIL
    service = GymService()
    REPORT.update({'python': platform.python_version(), 'pillow': PIL.__version__, 'opencv': cv2.__version__, 'numpy': numpy.__version__, 'toolbox_source': str(service.source), 'toolbox_class_source': inspect.getfile(service.Toolbox), 'toolbox_digest': service.digest})
    run = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ToolboxBridgeTests))
    REPORT.update({'tests': run.testsRun, 'failures': len(run.failures), 'errors': len(run.errors), 'success': run.wasSuccessful()})
    print(json.dumps({'comparisons': len(REPORT['comparisons']), 'success': run.wasSuccessful()}))
    raise SystemExit(not run.wasSuccessful())
