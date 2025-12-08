**Random Forest**

**rfReport**

**Random Forest Confusion Matrix:**

![A diagram of a confusion matrix AI-generated content may be
incorrect.](media/image1.png){width="2.816208442694663in"
height="2.516344050743657in"}

**Random Forest Agreggated Peformance Metrics:**

**Random Forest classification_report:**

  --------------------------------------------------------------------------
          **Metric**                                     **Value**
  ------- ---------------------------------------------- -------------------
  **0**   Precision (Macro)                              0.980598

  **1**   Recall (Macro)                                 0.899422

  **2**   F1 Score (Macro)                               0.935151

  **3**   Precision (Weighted)                           0.982543

  **4**   Recall (Weighted)                              0.982478

  **5**   F1 Score (Weighted)                            0.981716

  **6**   Accuracy                                       0.982478

  **7**   Overall Model Accuracy                         0.982478
  --------------------------------------------------------------------------

**Overall Model Accuracy : 0.9824780976220275**

![A graph of a triangle AI-generated content may be
incorrect.](media/image2.png){width="9.184397419072615in"
height="3.038543307086614in"}

**Gradient Boosting**

![A diagram of a confusion matrix AI-generated content may be
incorrect.](media/image3.png){width="2.8150995188101486in"
height="2.515353237095363in"}

**Gradient Boosting Agreggated Peformance Metrics:**

  --------------------------------------------------------------------------
          **Metric**                                     **Value**
  ------- ---------------------------------------------- -------------------
  **0**   Precision (Macro)                              0.993033

  **1**   Recall (Macro)                                 0.917130

  **2**   F1 Score (Macro)                               0.951187

  **3**   Precision (Weighted)                           0.987654

  **4**   Recall (Weighted)                              0.987484

  **5**   F1 Score (Weighted)                            0.986939

  **6**   Accuracy                                       0.987484

  **7**   Overall Model Accuracy                         0.987484
  --------------------------------------------------------------------------

**Overall Model Accuracy : 0.9874843554443054**

![A graph of a triangle AI-generated content may be
incorrect.](media/image4.png){width="8.711199693788277in"
height="2.8819925634295713in"}

**Isolation Forest**

Isolation Forest classification_report:

**Isolation Forest Confusion Matrix:**

/usr/local/lib/python3.12/dist-packages/sklearn/metrics/\_classification.py:1565:
Undefined Metric Warning: Precision is ill-defined and being set to 0.0
in labels with no predicted samples. Use \`zero division\` parameter to
control this behavior.

\_warn_prf(average, modifier, f\"{metric.capitalize()} is\",
len(result))

/usr/local/lib/python3.12/dist-packages/sklearn/metrics/\_classification.py:1565:
Undefined Metric Warning: Precision is ill-defined and being set to 0.0
in labels with no predicted samples. Use \`zero division\` parameter to
control this behavior.

\_warn_prf(average, modifier, f\"{metric.capitalize()} is\",
len(result))

/usr/local/lib/python3.12/dist-packages/sklearn/metrics/\_classification.py:1565:
Undefined Metric Warning: Precision is ill-defined and being set to 0.0
in labels with no predicted samples. Use \`zero division\` parameter to
control this behavior.

\_warn_prf(average, modifier, f\"{metric.capitalize()} is\",
len(result))

![A chart with different colored squares AI-generated content may be
incorrect.](media/image5.png){width="3.7305555555555556in"
height="3.3333333333333335in"}

**Isolation Forest Agreggated Peformance Metrics:**

  --------------------------------------------------------------------------
          **Metric**                                     **Value**
  ------- ---------------------------------------------- -------------------
  **0**   Precision (Macro)                              0.147892

  **1**   Recall (Macro)                                 0.229550

  **2**   F1 Score (Macro)                               0.179888

  **3**   Precision (Weighted)                           0.362048

  **4**   Recall (Weighted)                              0.561952

  **5**   F1 Score (Weighted)                            0.440376

  **6**   Accuracy                                       0.561952

  **7**   Overall Model Accuracy                         0.561952
  --------------------------------------------------------------------------

Overall Model Accuracy : 0.5619524405506884

/usr/local/lib/python3.12/dist-packages/sklearn/metrics/\_ranking.py:1188:
UndefinedMetricWarning: No positive samples in y_true, true positive
value should be meaningless

warnings.warn(

/usr/local/lib/python3.12/dist-packages/sklearn/metrics/\_ranking.py:1033:
UserWarning: No positive class found in y_true, recall is set to one for
all thresholds.

warnings.warn(

![A graph with a line AI-generated content may be
incorrect.](media/image6.png){width="9.403324584426947in"
height="3.1109733158355204in"}

Epoch 1/100

/usr/local/lib/python3.12/dist-packages/keras/src/layers/core/dense.py:93:
UserWarning: Do not pass an \`input_shape\`/\`input_dim\` argument to a
layer. When using Sequential models, prefer using an \`Input(shape)\`
object as the first layer in the model instead.

super().\_\_init\_\_(activity_regularizer=activity_regularizer,
\*\*kwargs)

**90/90** ━━━━━━━━━━━━━━━━━━━━ **2s** 5ms/step - loss: 0.1548 -
val_loss: 0.1134

Epoch 2/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0912 -
val_loss: 0.0750

Epoch 3/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0679 -
val_loss: 0.0639

Epoch 4/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0600 -
val_loss: 0.0603

Epoch 5/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0565 -
val_loss: 0.0581

Epoch 6/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0570 -
val_loss: 0.0562

Epoch 7/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0530 -
val_loss: 0.0549

Epoch 8/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 5ms/step - loss: 0.0503 -
val_loss: 0.0540

Epoch 9/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **1s** 5ms/step - loss: 0.0514 -
val_loss: 0.0531

Epoch 10/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 5ms/step - loss: 0.0488 -
val_loss: 0.0521

Epoch 11/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 5ms/step - loss: 0.0488 -
val_loss: 0.0511

Epoch 12/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 5ms/step - loss: 0.0485 -
val_loss: 0.0501

Epoch 13/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 5ms/step - loss: 0.0476 -
val_loss: 0.0493

Epoch 14/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 4ms/step - loss: 0.0463 -
val_loss: 0.0489

Epoch 15/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0463 -
val_loss: 0.0487

Epoch 16/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0457 -
val_loss: 0.0484

Epoch 17/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0455 -
val_loss: 0.0483

Epoch 18/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0474 -
val_loss: 0.0481

Epoch 19/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0481 -
val_loss: 0.0479

Epoch 20/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0461 -
val_loss: 0.0478

Epoch 21/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0459 -
val_loss: 0.0479

Epoch 22/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0458 -
val_loss: 0.0477

Epoch 23/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0459 -
val_loss: 0.0475

Epoch 24/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0467 -
val_loss: 0.0473

Epoch 25/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0436 -
val_loss: 0.0471

Epoch 26/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0448 -
val_loss: 0.0472

Epoch 27/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0442 -
val_loss: 0.0470

Epoch 28/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0442 -
val_loss: 0.0469

Epoch 29/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0437 -
val_loss: 0.0469

Epoch 30/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0435 -
val_loss: 0.0468

Epoch 31/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0430 -
val_loss: 0.0469

Epoch 32/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0447 -
val_loss: 0.0466

Epoch 33/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0450 -
val_loss: 0.0468

Epoch 34/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0431 -
val_loss: 0.0465

Epoch 35/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0414 -
val_loss: 0.0464

Epoch 36/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0431 -
val_loss: 0.0463

Epoch 37/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0440 -
val_loss: 0.0463

Epoch 38/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0443 -
val_loss: 0.0463

Epoch 39/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0433 -
val_loss: 0.0462

Epoch 40/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0445 -
val_loss: 0.0462

Epoch 41/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0455 -
val_loss: 0.0461

Epoch 42/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0447 -
val_loss: 0.0460

Epoch 43/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0440 -
val_loss: 0.0461

Epoch 44/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0429 -
val_loss: 0.0460

Epoch 45/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0441 -
val_loss: 0.0459

Epoch 46/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0431 -
val_loss: 0.0460

Epoch 47/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0441 -
val_loss: 0.0459

Epoch 48/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 4ms/step - loss: 0.0421 -
val_loss: 0.0457

Epoch 49/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 4ms/step - loss: 0.0425 -
val_loss: 0.0458

Epoch 50/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **1s** 4ms/step - loss: 0.0431 -
val_loss: 0.0456

Epoch 51/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 4ms/step - loss: 0.0451 -
val_loss: 0.0456

Epoch 52/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 5ms/step - loss: 0.0418 -
val_loss: 0.0458

Epoch 53/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 5ms/step - loss: 0.0450 -
val_loss: 0.0456

Epoch 54/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0434 -
val_loss: 0.0456

Epoch 55/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0420 -
val_loss: 0.0454

Epoch 56/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0446 -
val_loss: 0.0455

Epoch 57/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0413 -
val_loss: 0.0455

Epoch 58/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0449 -
val_loss: 0.0454

Epoch 59/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0425 -
val_loss: 0.0452

Epoch 60/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0443 -
val_loss: 0.0454

Epoch 61/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0451 -
val_loss: 0.0454

Epoch 62/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0414 -
val_loss: 0.0453

Epoch 63/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0436 -
val_loss: 0.0453

Epoch 64/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0429 -
val_loss: 0.0451

Epoch 65/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0444 -
val_loss: 0.0452

Epoch 66/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0425 -
val_loss: 0.0451

Epoch 67/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0417 -
val_loss: 0.0450

Epoch 68/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0429 -
val_loss: 0.0450

Epoch 69/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0431 -
val_loss: 0.0450

Epoch 70/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0439 -
val_loss: 0.0450

Epoch 71/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0426 -
val_loss: 0.0449

Epoch 72/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0426 -
val_loss: 0.0450

Epoch 73/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0430 -
val_loss: 0.0449

Epoch 74/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0415 -
val_loss: 0.0450

Epoch 75/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0419 -
val_loss: 0.0449

Epoch 76/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0417 -
val_loss: 0.0448

Epoch 77/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0437 -
val_loss: 0.0451

Epoch 78/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0427 -
val_loss: 0.0448

Epoch 79/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0434 -
val_loss: 0.0447

Epoch 80/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0437 -
val_loss: 0.0448

Epoch 81/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0450 -
val_loss: 0.0447

Epoch 82/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **1s** 3ms/step - loss: 0.0412 -
val_loss: 0.0449

Epoch 83/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0430 -
val_loss: 0.0448

Epoch 84/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0433 -
val_loss: 0.0447

Epoch 85/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0418 -
val_loss: 0.0446

Epoch 86/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0422 -
val_loss: 0.0447

Epoch 87/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 4ms/step - loss: 0.0405 -
val_loss: 0.0448

Epoch 88/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 4ms/step - loss: 0.0418 -
val_loss: 0.0446

Epoch 89/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **1s** 5ms/step - loss: 0.0422 -
val_loss: 0.0446

Epoch 90/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 5ms/step - loss: 0.0423 -
val_loss: 0.0446

Epoch 91/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **1s** 5ms/step - loss: 0.0410 -
val_loss: 0.0447

Epoch 92/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0434 -
val_loss: 0.0446

Epoch 93/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0419 -
val_loss: 0.0448

Epoch 94/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0411 -
val_loss: 0.0445

Epoch 95/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0432 -
val_loss: 0.0447

Epoch 96/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0446 -
val_loss: 0.0446

Epoch 97/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0438 -
val_loss: 0.0445

Epoch 98/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **1s** 3ms/step - loss: 0.0433 -
val_loss: 0.0445

Epoch 99/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0422 -
val_loss: 0.0445

Epoch 100/100

**90/90** ━━━━━━━━━━━━━━━━━━━━ **0s** 3ms/step - loss: 0.0409 -
val_loss: 0.0445

**25/25** ━━━━━━━━━━━━━━━━━━━━ **0s** 4ms/step

Auto encoder

autoencoderReport

Autoencoder classification_report:

Autoencoder Confusion Matrix:

/usr/local/lib/python3.12/dist-packages/sklearn/metrics/\_classification.py:1565:
UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in
labels with no predicted samples. Use \`zero_division\` parameter to
control this behavior.

\_warn_prf(average, modifier, f\"{metric.capitalize()} is\",
len(result))

/usr/local/lib/python3.12/dist-packages/sklearn/metrics/\_classification.py:1565:
UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in
labels with no predicted samples. Use \`zero_division\` parameter to
control this behavior.

\_warn_prf(average, modifier, f\"{metric.capitalize()} is\",
len(result))

/usr/local/lib/python3.12/dist-packages/sklearn/metrics/\_classification.py:1565:
UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in
labels with no predicted samples. Use \`zero_division\` parameter to
control this behavior.

\_warn_prf(average, modifier, f\"{metric.capitalize()} is\",
len(result))

![A chart with different colored squares AI-generated content may be
incorrect.](media/image5.png){width="3.7305555555555556in"
height="3.3333333333333335in"}

Autoencoder Agreggated Peformance Metrics:

  --------------------------------------------------------------------------
          **Metric**                                     **Value**
  ------- ---------------------------------------------- -------------------
  **0**   Precision (Macro)                              0.147892

  **1**   Recall (Macro)                                 0.229550

  **2**   F1 Score (Macro)                               0.179888

  **3**   Precision (Weighted)                           0.362048

  **4**   Recall (Weighted)                              0.561952

  **5**   F1 Score (Weighted)                            0.440376

  **6**   Accuracy                                       0.561952

  **7**   Overall Model Accuracy                         0.561952
  --------------------------------------------------------------------------

Overall Model Accuracy : 0.5619524405506884

![A graph with a triangle AI-generated content may be
incorrect.](media/image7.png){width="9.174833770778653in"
height="3.0353794838145234in"}

OneClassSVM

OneClassSVMReport

one_class_svm classification_report:

one_class_svm Confusion Matrix:

/usr/local/lib/python3.12/dist-packages/sklearn/metrics/\_classification.py:1565:
UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in
labels with no predicted samples. Use \`zero_division\` parameter to
control this behavior.

\_warn_prf(average, modifier, f\"{metric.capitalize()} is\",
len(result))

/usr/local/lib/python3.12/dist-packages/sklearn/metrics/\_classification.py:1565:
UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in
labels with no predicted samples. Use \`zero_division\` parameter to
control this behavior.

\_warn_prf(average, modifier, f\"{metric.capitalize()} is\",
len(result))

/usr/local/lib/python3.12/dist-packages/sklearn/metrics/\_classification.py:1565:
UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in
labels with no predicted samples. Use \`zero_division\` parameter to
control this behavior.

\_warn_prf(average, modifier, f\"{metric.capitalize()} is\",
len(result))

![A chart with different colored squares AI-generated content may be
incorrect.](media/image8.png){width="3.7305555555555556in"
height="3.3333333333333335in"}

one_class_svm Agreggated Peformance Metrics:

  --------------------------------------------------------------------------
          **Metric**                                     **Value**
  ------- ---------------------------------------------- -------------------
  **0**   Precision (Macro)                              0.148355

  **1**   Recall (Macro)                                 0.230573

  **2**   F1 Score (Macro)                               0.180544

  **3**   Precision (Weighted)                           0.363183

  **4**   Recall (Weighted)                              0.564456

  **5**   F1 Score (Weighted)                            0.441984

  **6**   Accuracy                                       0.564456

  **7**   Overall Model Accuracy                         0.564456
  --------------------------------------------------------------------------

Overall Model Accuracy : 0.5644555694618273

![A graph with a triangle AI-generated content may be
incorrect.](media/image9.png){width="8.983853893263342in"
height="2.972195975503062in"}

/usr/local/lib/python3.12/dist-packages/sklearn/utils/validation.py:2739:
UserWarning: X does not have valid feature names, but LocalOutlierFactor
was fitted with feature names

warnings.warn(

Local Outlier Factor

lofReport

/usr/local/lib/python3.12/dist-packages/sklearn/metrics/\_classification.py:1565:
UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in
labels with no predicted samples. Use \`zero_division\` parameter to
control this behavior.

\_warn_prf(average, modifier, f\"{metric.capitalize()} is\",
len(result))

/usr/local/lib/python3.12/dist-packages/sklearn/metrics/\_classification.py:1565:
UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in
labels with no predicted samples. Use \`zero_division\` parameter to
control this behavior.

\_warn_prf(average, modifier, f\"{metric.capitalize()} is\",
len(result))

/usr/local/lib/python3.12/dist-packages/sklearn/metrics/\_classification.py:1565:
UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in
labels with no predicted samples. Use \`zero_division\` parameter to
control this behavior.

\_warn_prf(average, modifier, f\"{metric.capitalize()} is\",
len(result))

Local Outlier Factor classification_report:

Local Outlier Factor Confusion Matrix:

![](media/image10.png){width="3.7305555555555556in"
height="3.3333333333333335in"}

**Local Outlier Factor Aggregate Performance Metrics:**

  --------------------------------------------------------------------------
          **Metric**                                     **Value**
  ------- ---------------------------------------------- -------------------
  **0**   Precision (Macro)                              0.187549

  **1**   Recall (Macro)                                 0.316268

  **2**   F1 Score (Macro)                               0.231867

  **3**   Precision (Weighted)                           0.388056

  **4**   Recall (Weighted)                              0.575720

  **5**   F1 Score (Weighted)                            0.462948

  **6**   Accuracy                                       0.575720

  **7**   Overall Model Accuracy                         0.575720
  --------------------------------------------------------------------------

Overall Model Accuracy : 0.5757196495619524

![A graph with a triangle AI-generated content may be
incorrect.](media/image11.png){width="9.254654418197726in"
height="3.061788057742782in"}

**Density-Based Spatial Clustering of Applications with Noise(DBSCAN)**

**dbscanReport**

**DBSCAN classification_report:**

**DBSCAN Confusion Matrix:**

/usr/local/lib/python3.12/dist-packages/sklearn/metrics/\_classification.py:1565:
UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in
labels with no predicted samples. Use \`zero_division\` parameter to
control this behavior.

\_warn_prf(average, modifier, f\"{metric.capitalize()} is\",
len(result))

/usr/local/lib/python3.12/dist-packages/sklearn/metrics/\_classification.py:1565:
UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in
labels with no predicted samples. Use \`zero_division\` parameter to
control this behavior.

\_warn_prf(average, modifier, f\"{metric.capitalize()} is\",
len(result))

/usr/local/lib/python3.12/dist-packages/sklearn/metrics/\_classification.py:1565:
UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in
labels with no predicted samples. Use \`zero_division\` parameter to
control this behavior.

\_warn_prf(average, modifier, f\"{metric.capitalize()} is\",
len(result))

![A chart with different colored squares AI-generated content may be
incorrect.](media/image12.png){width="3.098788276465442in"
height="2.7688363954505686in"}

**DBSCAN Aggregate Performance Metrics:**

  --------------------------------------------------------------------------
          **Metric**                                     **Value**
  ------- ---------------------------------------------- -------------------
  **0**   Precision (Macro)                              0.122951

  **1**   Recall (Macro)                                 0.153374

  **2**   F1 Score (Macro)                               0.136488

  **3**   Precision (Weighted)                           0.300991

  **4**   Recall (Weighted)                              0.375469

  **5**   F1 Score (Weighted)                            0.334130

  **6**   Accuracy                                       0.375469

  **7**   Overall Model Accuracy                         0.375469
  --------------------------------------------------------------------------

**Overall Model Accuracy : 0.37546933667083854**

![A graph with a triangle AI-generated content may be
incorrect.](media/image13.png){width="9.047670603674542in"
height="2.9933081802274715in"}

**lstmReport**

**LSTM classification_report:**

**LSTM Confusion Matrix:**

/usr/local/lib/python3.12/dist-packages/sklearn/metrics/\_classification.py:1565:
UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in
labels with no predicted samples. Use \`zero_division\` parameter to
control this behavior.

\_warn_prf(average, modifier, f\"{metric.capitalize()} is\",
len(result))

/usr/local/lib/python3.12/dist-packages/sklearn/metrics/\_classification.py:1565:
UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in
labels with no predicted samples. Use \`zero_division\` parameter to
control this behavior.

\_warn_prf(average, modifier, f\"{metric.capitalize()} is\",
len(result))

/usr/local/lib/python3.12/dist-packages/sklearn/metrics/\_classification.py:1565:
UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in
labels with no predicted samples. Use \`zero_division\` parameter to
control this behavior.

\_warn_prf(average, modifier, f\"{metric.capitalize()} is\",
len(result))

![A chart with different colored squares AI-generated content may be
incorrect.](media/image5.png){width="3.7305555555555556in"
height="3.3333333333333335in"}

L

**STM Agreggated Peformance Metrics:**

  --------------------------------------------------------------------------
          **Metric**                                     **Value**
  ------- ---------------------------------------------- -------------------
  **0**   Precision (Macro)                              0.147892

  **1**   Recall (Macro)                                 0.229550

  **2**   F1 Score (Macro)                               0.179888

  **3**   Precision (Weighted)                           0.362048

  **4**   Recall (Weighted)                              0.561952

  **5**   F1 Score (Weighted)                            0.440376

  **6**   Accuracy                                       0.561952

  **7**   Overall Model Accuracy                         0.561952
  --------------------------------------------------------------------------

**Overall Model Accuracy : 0.5619524405506884**

![A graph with a triangle AI-generated content may be
incorrect.](media/image14.png){width="8.949725503062117in"
height="2.960904418197725in"}

**K-Means**

**kmeansReport**

**k-means classification_report:**

**k-means Confusion Matrix:**

/usr/local/lib/python3.12/dist-packages/sklearn/metrics/\_classification.py:1565:
UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in
labels with no predicted samples. Use \`zero_division\` parameter to
control this behavior.

\_warn_prf(average, modifier, f\"{metric.capitalize()} is\",
len(result))

/usr/local/lib/python3.12/dist-packages/sklearn/metrics/\_classification.py:1565:
UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in
labels with no predicted samples. Use \`zero_division\` parameter to
control this behavior.

\_warn_prf(average, modifier, f\"{metric.capitalize()} is\",
len(result))

/usr/local/lib/python3.12/dist-packages/sklearn/metrics/\_classification.py:1565:
UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in
labels with no predicted samples. Use \`zero_division\` parameter to
control this behavior.

\_warn_prf(average, modifier, f\"{metric.capitalize()} is\",
len(result))

![](media/image15.png){width="3.7305555555555556in"
height="3.3333333333333335in"}

**k-means Agreggated Peformance Metrics:**

  --------------------------------------------------------------------------
          **Metric**                                     **Value**
  ------- ---------------------------------------------- -------------------
  **0**   Precision (Macro)                              0.261924

  **1**   Recall (Macro)                                 0.347648

  **2**   F1 Score (Macro)                               0.163204

  **3**   Precision (Weighted)                           0.613746

  **4**   Recall (Weighted)                              0.275344

  **5**   F1 Score (Weighted)                            0.347113

  **6**   Accuracy                                       0.275344

  **7**   Overall Model Accuracy                         0.275344
  --------------------------------------------------------------------------

**Overall Model Accuracy : 0.2753441802252816**

![A graph with a triangle AI-generated content may be
incorrect.](media/image16.png){width="9.091061898512686in"
height="3.007665135608049in"}

**Best performing model: RandomForestClassifier(random_state=42)**

**Best model metric: 0**

**{\'Precision (Macro)\': 0.9805981148785425,**

**\'Recall (Macro)\': 0.8994224439333867,**

**\'F1 Score (Macro)\': 0.9351507020311478,**

**\'Precision (Weighted)\': 0.9825425221557311,**

**\'Recall (Weighted)\': 0.9824780976220275,**

**\'F1 Score (Weighted)\': 0.9817156263702216,**

**\'Accuracy\': 0.9824780976220275,**

**\'Overall Model Accuracy \': 0.9824780976220275}**

**Best model saved to:
CyberThreat_Insight/model_deployment/RandomForest_best_model.pkl**

**\<Figure size 640x480 with 0 Axes\>**
