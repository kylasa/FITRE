
#include <functions/cnn_eval_model.h>

#include <functions/cnn_forward.h>


real evaluateCNNModel (CNN_MODEL *model, DEVICE_DATASET *data, 
	SCRATCH_AREA *scratch, real *z, real *probs, real *lossFuncErrors, 
	int offset, int curBatchSize)
{
	 return cnnForward( model, data, scratch, z, probs, 
					lossFuncErrors, offset, curBatchSize, MODEL_TRAIN ); 
}
