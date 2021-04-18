from flask import Flask
import os
from nn_evaluate import evaluate
from data import get_finnhub_data
from data import parse_real_time_data

app = Flask(__name__)

x = [24866.94921875, 24900.330078125, 24979.640625, 25195.470703125, 25237.689453125, 25295.330078125, 25407.0, 25450.0, 25440.560546875, 25506.76953125, 25609.220703125, 25716.23046875, 25715.279296875, 25766.779296875, 25899.0, 25950.0, 25807.439453125, 25761.279296875, 25787.55078125, 25774.869140625, 25738.08984375, 25694.7890625, 25648.939453125, 25683.83984375, 25743.76953125, 25852.279296875, 25969.619140625, 25854.48046875, 25926.58984375, 26255.509765625, 26332.970703125, 26472.5, 26463.439453125, 26776.580078125, 26686.759765625, 26426.560546875, 26378.650390625, 26439.01953125, 26400.0, 26524.150390625, 26531.4296875, 26401.490234375, 26521.169921875, 26584.619140625, 26648.2109375, 26695.189453125, 26705.970703125, 26887.69921875, 26777.220703125, 26645.91015625, 26607.759765625, 26750.51953125, 26709.7890625, 26737.109375, 26620.630859375, 26632.0703125, 26616.58984375, 26484.400390625, 26610.599609375, 26712.689453125, 26729.609375, 26711.029296875, 26710.470703125, 26777.759765625, 26774.310546875, 26768.7109375, 26968.5390625, 27369.33984375, 27517.8203125, 27561.150390625, 27587.810546875, 27535.7109375, 27658.080078125, 27626.5, 27654.640625, 27871.439453125, 27740.0, 27789.990234375, 27200.0, 27687.240234375, 27686.630859375, 27646.2890625, 27771.890625, 27745.33984375, 27707.7109375, 27730.58984375, 27788.470703125, 28319.41015625, 28253.7890625, 27559.640625, 27500.0, 26970.220703125, 26849.5703125, 27262.900390625, 27252.490234375, 27295.609375, 27145.05078125, 27373.26953125, 27489.30078125, 27726.630859375]

y = [27549.646484375, 27482.6328125, 27422.19140625, 27392.548828125, 27473.203125, 27459.408203125, 27570.298828125, 27661.69921875, 27569.44921875, 27631.404296875, 27723.064453125, 27698.142578125, 27732.99609375, 27735.57421875, 27751.4921875, 27622.490234375, 27609.296875, 27489.486328125, 27551.7578125, 27455.248046875, 27465.5078125, 27394.091796875, 27390.208984375, 27453.009765625, 27388.77734375, 27400.146484375, 27302.80078125, 27366.322265625, 27278.748046875, 27209.1328125, 27225.19921875, 27192.908203125, 27195.90625, 27091.177734375, 27049.791015625, 27043.611328125, 27088.1640625, 27236.794921875, 27177.47265625, 27164.2109375, 27147.908203125, 27182.970703125, 27199.04296875, 27222.560546875, 27186.36328125, 27113.521484375, 27209.91015625, 27242.763671875, 27187.84375, 27296.783203125, 27331.5390625, 27425.0234375, 27514.82421875, 27466.75390625, 27426.53125, 27455.80078125, 27445.94921875, 27440.451171875, 27503.427734375, 27623.05078125, 27517.220703125, 27509.080078125, 27454.349609375, 27461.517578125, 27494.69921875, 27566.728515625, 27405.30078125, 27461.69140625, 27438.57421875, 27420.150390625, 27437.84765625, 27298.21484375, 27322.8828125, 27290.2265625, 27256.291015625, 27295.021484375, 27210.31640625, 27207.3203125, 27177.00390625, 27279.115234375, 27233.609375, 27171.1640625, 27225.12109375, 27280.306640625, 27235.67578125, 27123.251953125, 27165.541015625, 27279.65625, 27137.009765625, 27181.7890625, 27311.861328125, 27424.578125, 27463.11328125, 27498.55078125, 27459.1328125, 27397.9296875, 27416.07421875, 27356.640625, 27370.16015625, 27394.873046875]

@app.route('/')
def hello_world():
    current_data = get_finnhub_data()
    model_input = parse_real_time_data(current_data)
    output = evaluate(model_input)

    return {
        'current': x,
        'prediction': output
    }

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run()

