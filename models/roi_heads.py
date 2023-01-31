import torch
import torch.nn.functional as F
from torch import nn, Tensor

from torchvision.ops import boxes as box_ops
from torchvision.models.detection import _utils as det_utils

from typing import Optional, List, Dict, Tuple
from .rcnn_loss import *


class RoIHeads(nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(self,
                 box_roi_pool,
                 box_head,
                 box_predictor,
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detections_per_img,
                 # Mask
                 mask_roi_pool=None,
                 mask_head=None,
                 mask_predictor=None,
                 keypoint_roi_pool=None,
                 keypoint_head=None,
                 keypoint_predictor=None,
                 num_kps3d=50,
                 num_verts=2556
                 ):
        super(RoIHeads, self).__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,
            positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor

        self.keypoint_roi_pool = keypoint_roi_pool
        self.keypoint_head = keypoint_head
        self.keypoint_predictor = keypoint_predictor
        
        self.MAX_SIZE = 1400

    def has_keypoint(self):
        if self.keypoint_roi_pool is None:
            return False
        if self.keypoint_head is None:
            return False
        if self.keypoint_predictor is None:
            return False
        return True

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):

            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]]) -> None
        assert targets is not None
        assert all(["boxes" in t for t in targets])
        assert all(["labels" in t for t in targets])

    def select_training_samples(self,
                                proposals,  # type: List[Tensor]
                                targets     # type: Optional[List[Dict[str, Tensor]]]
                                ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        self.check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(self,
                               class_logits,    # type: Tensor
                               box_regression,  # type: Tensor
                               proposals,       # type: List[Tensor]
                               image_shapes     # type: List[Tuple[int, int]]
                               ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(self,
                features,      # type: Dict[str, Tensor]
                proposals,     # type: List[Tensor]
                image_shapes,  # type: List[Tuple[int, int]]
                original_imgs, # type: List[Tensor]
                targets=None   # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        
        class_logits, box_regression = self.box_predictor(box_features)
        result: List[Dict[str, torch.Tensor]] = []
        losses = {}

        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )
        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if self.keypoint_roi_pool is not None and self.keypoint_head is not None \
                and self.keypoint_predictor is not None:
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                assert matched_idxs is not None
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None
            
            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            graformer_features = keypoint_features
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            # Apply the same operation but only the last RoIs which are the GT RoIs to train the GraFormers
            # TODO: extend the filter operation to create multiple 3-roi samples
            batch=0
            if self.num_classes > 2:
                filtered_keypoint_proposals = filter_rois(keypoint_proposals, self.training, labels)
            else:
                batch, kps, H, W = keypoint_logits.shape
                filtered_keypoint_proposals = None

            keypoint3d = torch.zeros((len(proposals), self.num_kps3d, 3))
            mesh3d = torch.zeros((len(proposals), self.num_verts, 3))

            # There are two modes to use RCNN's outputs to reconstruct hands and object 
            # If the hand and object in the same box i.e. num_classes=2, or if the hands and the objects are in separate boxes i.e. num_classes = 3 or 4

            if self.num_verts > 0 and ((self.num_classes > 2 and filtered_keypoint_proposals is not None) or batch > 0):
                
                if self.num_classes > 2: 
                    graformer_features = self.keypoint_roi_pool(features, filtered_keypoint_proposals, image_shapes)
                    keypoint_features = self.keypoint_head(graformer_features)
                    graformer_keypoint_logits = self.keypoint_predictor(keypoint_features)
                else:
                    graformer_keypoint_logits = keypoint_logits

                if self.graph_input == 'heatmaps':
                    batch, kps, H, W = graformer_keypoint_logits.shape                   
                    graformer_inputs = graformer_keypoint_logits.view(batch, kps, W * H)
                else:
                    # Convert heatmaps and RoIs to 2D points
                    keypoints_probs, _ = keypointrcnn_inference(graformer_keypoint_logits, filtered_keypoint_proposals)
                    keypoints2d = torch.stack(keypoints_probs, dim=0)
                    batch, num_classes, kps, dimension = keypoints2d.shape
                    graformer_inputs = keypoints2d.view(batch, num_classes * kps, dimension)[:, :self.num_kps3d, :2]                
                
                # Estimate 3D pose
                keypoint3d = self.keypoint_graformer(graformer_inputs)
                
                # Extract features from RoIs
                graformer_features = self.feature_extractor(graformer_features)
                
                # Every image has 3 RoIs 
                if self.num_classes > 2:
                    graformer_features = graformer_features.view(num_images, num_classes, -1).unsqueeze(axis=2).repeat(1, 1, self.num_kps2d, 1)
                    graformer_features = graformer_features.view(num_images, num_classes * kps, 2048)[:, :self.num_kps3d]
                else:
                    graformer_features = graformer_features.unsqueeze(axis=1).repeat(1, self.num_kps2d, 1)
                
                # Pass features and pose to Coarse-to-fine GraFormer
                mesh_graformer_inputs = torch.cat((graformer_inputs, graformer_features), axis=2)
                mesh3d = self.mesh_graformer(mesh_graformer_inputs)
                hand_mesh = mesh3d[:, :778, :3]
                nimble_mesh, nimble_texture = self.mano_nimble(hand_mesh)

            loss_keypoint = {}
            
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None

                gt_keypoints = [t["keypoints"] for t in targets]
                keypoints3d_gt = [t["keypoints3d"] for t in targets]
                mesh3d_gt = [t["mesh3d"] for t in targets]

                # Shift back using palms
                if "palm" in targets[0].keys():
                    palms_gt = [t["palm"] for t in targets]
                else:
                    palms_gt = None

                rcnn_loss_keypoint, rcnn_loss_keypoint3d, rcnn_loss_mesh3d, rcnn_loss_photometric, nimble_loss = keypointrcnn_loss(
                                                                            keypoint_logits, keypoint_proposals, gt_keypoints, 
                                                                            pos_matched_idxs, keypoint3d, keypoints3d_gt, 
                                                                            mesh3d, mesh3d_gt, original_images=original_imgs, 
                                                                            palms_gt=palms_gt, num_classes=self.num_classes,
                                                                            photometric=self.photometric, dataset_name=self.dataset_name,
                                                                            nimble_mesh=nimble_mesh)

                loss_keypoint = {
                    "loss_keypoint": rcnn_loss_keypoint,
                    "loss_keypoint3d": rcnn_loss_keypoint3d,
                }
                
                if self.num_verts > 0:
                    loss_keypoint['loss_mesh3d'] = rcnn_loss_mesh3d
                    loss_keypoint['loss_nimble'] = nimble_loss
                if rcnn_loss_photometric is not None:
                    loss_keypoint['loss_photometric'] = rcnn_loss_photometric

            else:
                assert keypoint_logits is not None
                assert keypoint_proposals is not None

                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps
                    r["keypoints3d"] = keypoint3d
                    r["mesh3d"] = mesh3d        

            losses.update(loss_keypoint)

        return result, losses