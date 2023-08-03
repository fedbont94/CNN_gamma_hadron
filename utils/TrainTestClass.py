#!/usr/bin/env python3

import os
import sys
import time
import torch
import numpy as np
import torch.utils.data as data

from utils.utils_functions import load_data, make_input_tensors, format_duration


class TrainTestClass:
    def __init__(
        self,
        args,
        net,
        criterion,
        optimizer,
        scheduler,
        loadData=True,
    ):
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args

        if loadData:
            train_df = load_data(args=self.args, is_train=True)
            train_tensorDict = make_input_tensors(train_df)
            self.train_qMap_tensor = train_tensorDict["MapHLCq"]
            self.train_tMap_tensor = train_tensorDict["MapHLCt"]
            self.train_fccInput_tensor = train_tensorDict["fccInput"]
            self.train_output_tensor = train_tensorDict["output"]
            self.train_weights = train_tensorDict["weights"]

            test_df = load_data(args=self.args, is_train=False)
            test_tensorDict = make_input_tensors(test_df)
            self.test_qMap_tensor = test_tensorDict["MapHLCq"]
            self.test_tMap_tensor = test_tensorDict["MapHLCt"]
            self.test_fccInput_tensor = test_tensorDict["fccInput"]
            self.test_output_tensor = test_tensorDict["output"]
            self.test_weights = test_tensorDict["weights"]

        # Check if the job is running on an dev_accelerated partition
        self.partition_name = str(os.environ.get("SLURM_JOB_PARTITION"))

    def train(self, input_tensor):
        self.net.train()  # Set the network in training mode
        num_batches = len(input_tensor)

        total_loss = 0.0
        total_accuracy = 0.0

        for batch_idx, (qMap, tMap, fccInput, output_tensor, weights) in enumerate(
            input_tensor
        ):
            # start_idx = batch_idx * self.args.batchSize
            # end_idx = min(start_idx + self.args.batchSize, num_samples)

            self.optimizer.zero_grad()

            # Forward pass
            output = self.net(
                qMap=qMap,
                tMap=tMap,
                fcc=fccInput,
            )

            loss = self.criterion(output, output_tensor)

            accuracy = torch.mean(((output > 0.5).float() == output_tensor).float())

            weighted_loss = torch.mean(
                loss * weights
            )  # TODO: Fix weights only if not using Adam optimizer

            # Backward pass
            weighted_loss.backward()
            self.optimizer.step()

            # Show progress if the partition is dev_accelerated
            if self.partition_name == "dev_accelerated":
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

    def test(self, input_tensor):
        self.net.eval()  # Set the network in evaluation mode
        # Check the length of the input_tensor (DataLoader)
        if len(input_tensor) != 1:
            raise ValueError(
                "Expected input_tensor(DataLoader) to contain exactly one batch, "
                + f"but found {len(input_tensor)} batches."
            )

        # Get the first (and only) batch from the input_tensor (DataLoader)
        qMap, tMap, fccInput, output_tensor, weights = next(iter(input_tensor))

        with torch.no_grad():
            output = self.net(
                qMap=qMap,
                tMap=tMap,
                fcc=fccInput,
            )
            loss = self.criterion(output, output_tensor)

            weighted_loss = torch.mean(loss * weights).item()

            accuracy = torch.mean(
                ((output > 0.5).float() == output_tensor).float()
            ).item()

        return weighted_loss, accuracy, output

    def train_loop(self):
        print("Starting the training loop...")
        # Lists to store the training and testing losses and accuracies
        train_losses = []
        train_accuracies = []

        val_losses = []
        val_accuracies = []

        test_losses = []
        test_accuracies = []

        # Training loop
        for epoch in range(self.args.numEpochs):
            sys.stdout.flush()  # Flush the output after the loop

            # Start time of the current epoch
            epoch_start_time = time.time()

            # Create a random index permutation (optional) to shuffle the data before splitting
            indices = np.random.permutation(self.train_qMap_tensor.size(0))

            # Calculate the index to split the data
            # # 80% for training, 20% for validation
            split_idx = int(len(indices) * 0.8)

            # Split the data into training and validation sets
            train_indices, val_indices = indices[:split_idx], indices[split_idx:]

            # Create data loaders for training and validation
            train_sampler = data.SubsetRandomSampler(train_indices)
            val_sampler = data.SubsetRandomSampler(val_indices)

            train_loader = data.DataLoader(
                dataset=data.TensorDataset(
                    self.train_qMap_tensor,
                    self.train_tMap_tensor,
                    self.train_fccInput_tensor,
                    self.train_output_tensor,
                    self.train_weights,
                ),
                batch_size=self.args.batchSize,
                sampler=train_sampler,
            )

            val_loader = data.DataLoader(
                dataset=data.TensorDataset(
                    self.train_qMap_tensor,
                    self.train_tMap_tensor,
                    self.train_fccInput_tensor,
                    self.train_output_tensor,
                    self.train_weights,
                ),
                batch_size=len(val_indices),
                sampler=val_sampler,
            )

            # Test the entire test dataset in a single pass without shuffling or batching
            test_loader = data.DataLoader(
                dataset=data.TensorDataset(
                    self.test_qMap_tensor,
                    self.test_tMap_tensor,
                    self.test_fccInput_tensor,
                    self.test_output_tensor,
                    self.test_weights,
                ),
                # Set batch size to the size of the entire test dataset
                batch_size=self.test_qMap_tensor.size(0),
                shuffle=False,  # No shuffling
            )

            # Training
            train_loss, train_accuracy = self.train(train_loader)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            # Validation
            val_loss, val_accuracy, val_output = self.test(val_loader)
            # val_loss is used to reduce the learning rate
            self.scheduler.step(val_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            # Testing
            test_loss, test_accuracy, test_output = self.test(test_loader)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            # Calculate the duration of the current epoch
            epoch_duration = time.time() - epoch_start_time

            print(
                ""
                + f"Epoch: {epoch+1}/{self.args.numEpochs}, "
                #
                + f"Train Loss: {train_loss:.4f}, "
                + f"Val Loss: {val_loss:.4f}, "
                + f"Test Loss: {test_loss:.4f}, "
                #
                + f"Train Accuracy: {train_accuracy:.4f}, "
                + f"Val Accuracy: {val_accuracy:.4f}, "
                + f"Test Accuracy: {test_accuracy:.4f}, "
                #
                + f"Epoch Duration: {format_duration(int(epoch_duration))}"
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
                    "val_losses": val_losses,
                    "test_losses": test_losses,
                    #
                    "train_accuracies": train_accuracies,
                    "val_accuracies": val_accuracies,
                    "test_accuracies": test_accuracies,
                },
                f"{self.args.outputDir}/model/losses_accuracies.pth",
            )

            # Access the current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            # Stop training if the learning rate is too small
            if current_lr < 1e-9:
                print(
                    f"Learning rate is too small: {current_lr:.2e}. Stopping training..."
                )
                break

        print("Saving the model...")
        # Save the model
        torch.save(self.net.state_dict(), f"{self.args.outputDir}/model/model.pth")

        traning_results = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "test_losses": test_losses,
            #
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies,
            "test_accuracies": test_accuracies,
        }

        return traning_results
