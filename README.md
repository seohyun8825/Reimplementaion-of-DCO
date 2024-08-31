<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Direct Consistency Optimization (DCO) Reimplementation</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f4f4f4;
            color: #333;
        }

        .container {
            max-width: 800px;
            margin: auto;
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }

        h1 {
            color: #0066cc;
            text-align: center;
        }

        p {
            margin-bottom: 20px;
        }

        code {
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 4px;
            color: #d63384;
            font-size: 1.1em;
        }

        .note {
            background-color: #ffefc4;
            padding: 10px;
            border-left: 4px solid #ffcc00;
            border-radius: 4px;
        }

        .important {
            background-color: #e8f8e8;
            padding: 10px;
            border-left: 4px solid #28a745;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Direct Consistency Optimization (DCO) Reimplementation</h1>
        <p>This is a reimplementation of the Direct Consistency Optimization (DCO) paper.</p>

        <p>I’ve made some additional modifications to the DCO loss functions, which you can find in the <code>training/training_loop_</code> directory. There are several different versions of the loss function there.</p>

        <div class="important">
            <p>The most successful version is in <code>training_loop_fixed.py</code>. To run this code, rename the file to <code>training_loop.py</code>. I apologize for the messy implementation—I’ll tidy it up as soon as possible.</p>
        </div>

        <p>The idea behind the loss function in the <code>training_loop_fixed.py</code> file is to fully utilize the original DPO loss function (you can find this formula in the Diffusion for DPO paper).</p>

        <div class="note">
            <p>Your data will be set as personalized data, and the output of the fine-tuning model, before <code>n</code> steps, will be treated as the loss data. (This <code>n</code> is a hyperparameter, and experimentally, values between 50 and 100 worked best.)</p>
        </div>
    </div>
</body>
</html>
