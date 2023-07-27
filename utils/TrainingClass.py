#!/usr/bin/env python3

import os
import time
import torch

from utils.utils_functions import load_data


class TrainingClass:
    def __init__(self, args, net, criterion, optimizer, scheduler):
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args

        train_tensorDict = load_data(args=self.args, is_train=True)
        self.train_qMap_tensor = train_tensorDict["MapHLCq"]
        self.train_fccInput_tensor = train_tensorDict["fccInput"]
        self.train_output_tensor = train_tensorDict["output_tensor"]
        self.train_weights = train_tensorDict["weights"]

        test_tensorDict = load_data(args=self.args, is_train=False)
        self.test_qMap_tensor = test_tensorDict["MapHLCq"]
        self.test_fccInput_tensor = test_tensorDict["fccInput"]
        self.test_output_tensor = test_tensorDict["output_tensor"]
        self.test_weights = test_tensorDict["weights"]

    def train(self):
        self.net.train()  # Set the network in training mode
        num_samples = self.train_qMap_tensor.size(0)
        num_batches = (num_samples + self.args.batchSize - 1) // self.args.batchSize

        total_loss = 0.0
        total_accuracy = 0.0

        # Check if the job is running on an dev_accelerated partition
        partition_name = os.environ.get("SLURM_JOB_PARTITION")

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.args.batchSize
            end_idx = min(start_idx + self.args.batchSize, num_samples)

            self.optimizer.zero_grad()

            # Forward pass
            output = self.net(
                self.train_qMap_tensor[start_idx:end_idx],
                self.train_fccInput_tensor[start_idx:end_idx],
            )

            loss = self.criterion(output, self.train_output_tensor[start_idx:end_idx])

            accuracy = torch.mean(
                (
                    (output > 0.5).float()
                    == self.train_output_tensor[start_idx:end_idx]
                ).float()
            )

            weighted_loss = torch.mean(
                loss * self.train_weights[start_idx:end_idx]  # * 1e5
            )  # TODO: Fix weights only if not using Adam optimizer

            # Backward pass
            weighted_loss.backward()
            self.optimizer.step()

            # Show progress if the partition is dev_accelerated
            if str(partition_name) == "dev_accelerated":
                # Print progress
                progress = (batch_idx + 1) / num_batches * 100
                print(
                    f"Batch: [{batch_idx+1}/{num_batches}], Progress: {progress:.2f}%, Loss: {weighted_loss.item():.4f}",
                    end="\r",
                )

            total_loss += weighted_loss.item()
            total_accuracy += accuracy.item()

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches

        return avg_loss, avg_accuracy

    def test(self):
        with torch.no_grad():
            test_output = self.net(self.test_qMap_tensor, self.test_fccInput_tensor)
            loss = self.criterion(
                test_output[list(torch.isfinite(test_output))],
                self.test_output_tensor[list(torch.isfinite(test_output))],
            )

            weighted_loss = torch.mean(
                loss * self.test_weights[list(torch.isfinite(test_output))]
            ).item()

            accuracy = torch.mean(
                ((test_output > 0.5).float() == self.test_output_tensor).float()
            ).item()

        return weighted_loss, accuracy

    def train_loop(self):
        print("Starting the training loop...")
        # Lists to store the training and testing losses and accuracies
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []

        # Training loop
        for epoch in range(self.args.numEpochs):
            # Start time of the current epoch
            epoch_start_time = time.time()

            # Training
            train_loss, train_accuracy = self.train()
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            # Testing
            test_loss, test_accuracy = self.test()
            # test_loss is used to reduce the learning rate, but train loss is used bc no validation set
            self.scheduler.step(train_loss)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            # Calculate the duration of the current epoch
            epoch_duration = time.time() - epoch_start_time

            print(
                ""
                + f"Epoch: {epoch+1}/{self.args.numEpochs}, "
                + f"Train Loss: {train_loss:.4f}, "
                + f"Test Loss: {test_loss:.4f}, "
                + f"Train Accuracy: {train_accuracy:.4f}, "
                + f"Test Accuracy: {test_accuracy:.4f}, "
                + f"Epoch Duration: {epoch_duration:.2f} seconds"
            )
            # Save the model
            torch.save(
                self.net.state_dict(),
                f"{self.args.outputDir}/model/model_epoch{str(epoch).zfill(4)}.pth",
            )
            # Save the training and testing losses and accuracies
            torch.save(
                {
                    "train_losses": train_losses,
                    "test_losses": test_losses,
                    "train_accuracies": train_accuracies,
                    "test_accuracies": test_accuracies,
                },
                f"{self.args.outputDir}/model/losses_accuracies.pth",
            )

            # Stop training if the learning rate is too small
            if self.scheduler.get_lr()[0] < 1e-9:
                break

        print("Saving the model...")
        # Save the model
        torch.save(self.net.state_dict(), f"{self.args.outputDir}/model/model.pth")

        traning_results = {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "train_accuracies": train_accuracies,
            "test_accuracies": test_accuracies,
        }

        return traning_results
