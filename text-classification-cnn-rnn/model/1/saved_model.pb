��
�2�1
:
Add
x"T
y"T
z"T"
Ttype:
2	
�
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
E
AssignAddVariableOp
resource
value"dtype"
dtypetype�
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
B
Equal
x"T
y"T
z
"
Ttype:
2	
�
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
N
IsVariableInitialized
ref"dtype�
is_initialized
"
dtypetype�
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
�
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
;
Minimum
x"T
y"T
z"T"
Ttype:

2	�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	
�
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
L
PreventGradient

input"T
output"T"	
Ttype"
messagestring 
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
D
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
�
ResourceApplyAdam
var
m
v
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( �
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
#SparseSoftmaxCrossEntropyWithLogits
features"T
labels"Tlabels	
loss"T
backprop"T"
Ttype:
2"
Tlabelstype0	:
2	
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�
9
VarIsInitializedOp
resource
is_initialized
�
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.10.02b'v1.10.0-rc1-19-g656e7a2b34'��
~
Conv1_inputPlaceholder*
dtype0*/
_output_shapes
:���������*$
shape:���������
�
-Conv1/kernel/Initializer/random_uniform/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:*
_class
loc:@Conv1/kernel
�
+Conv1/kernel/Initializer/random_uniform/minConst*
valueB
 *HY��*
dtype0*
_output_shapes
: *
_class
loc:@Conv1/kernel
�
+Conv1/kernel/Initializer/random_uniform/maxConst*
valueB
 *HY�>*
dtype0*
_output_shapes
: *
_class
loc:@Conv1/kernel
�
5Conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniform-Conv1/kernel/Initializer/random_uniform/shape*
seed2 *
_class
loc:@Conv1/kernel*
T0*&
_output_shapes
:*
dtype0*

seed 
�
+Conv1/kernel/Initializer/random_uniform/subSub+Conv1/kernel/Initializer/random_uniform/max+Conv1/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *
_class
loc:@Conv1/kernel
�
+Conv1/kernel/Initializer/random_uniform/mulMul5Conv1/kernel/Initializer/random_uniform/RandomUniform+Conv1/kernel/Initializer/random_uniform/sub*
T0*&
_output_shapes
:*
_class
loc:@Conv1/kernel
�
'Conv1/kernel/Initializer/random_uniformAdd+Conv1/kernel/Initializer/random_uniform/mul+Conv1/kernel/Initializer/random_uniform/min*
T0*&
_output_shapes
:*
_class
loc:@Conv1/kernel
�
Conv1/kernelVarHandleOp*
shared_nameConv1/kernel*
_class
loc:@Conv1/kernel*
_output_shapes
: *
dtype0*
	container *
shape:
i
-Conv1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpConv1/kernel*
_output_shapes
: 
�
Conv1/kernel/AssignAssignVariableOpConv1/kernel'Conv1/kernel/Initializer/random_uniform*
dtype0*
_class
loc:@Conv1/kernel
�
 Conv1/kernel/Read/ReadVariableOpReadVariableOpConv1/kernel*
dtype0*&
_output_shapes
:*
_class
loc:@Conv1/kernel
�
Conv1/bias/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes
:*
_class
loc:@Conv1/bias
�

Conv1/biasVarHandleOp*
shared_name
Conv1/bias*
_class
loc:@Conv1/bias*
_output_shapes
: *
dtype0*
	container *
shape:
e
+Conv1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Conv1/bias*
_output_shapes
: 
{
Conv1/bias/AssignAssignVariableOp
Conv1/biasConv1/bias/Initializer/zeros*
dtype0*
_class
loc:@Conv1/bias
�
Conv1/bias/Read/ReadVariableOpReadVariableOp
Conv1/bias*
dtype0*
_output_shapes
:*
_class
loc:@Conv1/bias
d
Conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
p
Conv1/Conv2D/ReadVariableOpReadVariableOpConv1/kernel*
dtype0*&
_output_shapes
:
�
Conv1/Conv2DConv2DConv1_inputConv1/Conv2D/ReadVariableOp*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*/
_output_shapes
:���������*
use_cudnn_on_gpu(
c
Conv1/BiasAdd/ReadVariableOpReadVariableOp
Conv1/bias*
dtype0*
_output_shapes
:
�
Conv1/BiasAddBiasAddConv1/Conv2DConv1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*/
_output_shapes
:���������
[

Conv1/ReluReluConv1/BiasAdd*
T0*/
_output_shapes
:���������
W
flatten/ShapeShape
Conv1/Relu*
T0*
out_type0*
_output_shapes
:
e
flatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
g
flatten/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
g
flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
flatten/strided_sliceStridedSliceflatten/Shapeflatten/strided_slice/stackflatten/strided_slice/stack_1flatten/strided_slice/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask*
new_axis_mask *

begin_mask *
_output_shapes
: *
ellipsis_mask 
b
flatten/Reshape/shape/1Const*
valueB :
���������*
dtype0*
_output_shapes
: 
�
flatten/Reshape/shapePackflatten/strided_sliceflatten/Reshape/shape/1*

axis *
T0*
N*
_output_shapes
:
~
flatten/ReshapeReshape
Conv1/Reluflatten/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:����������

�
/Softmax/kernel/Initializer/random_uniform/shapeConst*
valueB"H  
   *
dtype0*
_output_shapes
:*!
_class
loc:@Softmax/kernel
�
-Softmax/kernel/Initializer/random_uniform/minConst*
valueB
 *7*
dtype0*
_output_shapes
: *!
_class
loc:@Softmax/kernel
�
-Softmax/kernel/Initializer/random_uniform/maxConst*
valueB
 *7�=*
dtype0*
_output_shapes
: *!
_class
loc:@Softmax/kernel
�
7Softmax/kernel/Initializer/random_uniform/RandomUniformRandomUniform/Softmax/kernel/Initializer/random_uniform/shape*
seed2 *!
_class
loc:@Softmax/kernel*
T0*
_output_shapes
:	�

*
dtype0*

seed 
�
-Softmax/kernel/Initializer/random_uniform/subSub-Softmax/kernel/Initializer/random_uniform/max-Softmax/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *!
_class
loc:@Softmax/kernel
�
-Softmax/kernel/Initializer/random_uniform/mulMul7Softmax/kernel/Initializer/random_uniform/RandomUniform-Softmax/kernel/Initializer/random_uniform/sub*
T0*
_output_shapes
:	�

*!
_class
loc:@Softmax/kernel
�
)Softmax/kernel/Initializer/random_uniformAdd-Softmax/kernel/Initializer/random_uniform/mul-Softmax/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
:	�

*!
_class
loc:@Softmax/kernel
�
Softmax/kernelVarHandleOp*
shared_nameSoftmax/kernel*!
_class
loc:@Softmax/kernel*
_output_shapes
: *
dtype0*
	container *
shape:	�


m
/Softmax/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpSoftmax/kernel*
_output_shapes
: 
�
Softmax/kernel/AssignAssignVariableOpSoftmax/kernel)Softmax/kernel/Initializer/random_uniform*
dtype0*!
_class
loc:@Softmax/kernel
�
"Softmax/kernel/Read/ReadVariableOpReadVariableOpSoftmax/kernel*
dtype0*
_output_shapes
:	�

*!
_class
loc:@Softmax/kernel
�
Softmax/bias/Initializer/zerosConst*
valueB
*    *
dtype0*
_output_shapes
:
*
_class
loc:@Softmax/bias
�
Softmax/biasVarHandleOp*
shared_nameSoftmax/bias*
_class
loc:@Softmax/bias*
_output_shapes
: *
dtype0*
	container *
shape:

i
-Softmax/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpSoftmax/bias*
_output_shapes
: 
�
Softmax/bias/AssignAssignVariableOpSoftmax/biasSoftmax/bias/Initializer/zeros*
dtype0*
_class
loc:@Softmax/bias
�
 Softmax/bias/Read/ReadVariableOpReadVariableOpSoftmax/bias*
dtype0*
_output_shapes
:
*
_class
loc:@Softmax/bias
m
Softmax/MatMul/ReadVariableOpReadVariableOpSoftmax/kernel*
dtype0*
_output_shapes
:	�


�
Softmax/MatMulMatMulflatten/ReshapeSoftmax/MatMul/ReadVariableOp*
T0*
transpose_b( *
transpose_a( *'
_output_shapes
:���������

g
Softmax/BiasAdd/ReadVariableOpReadVariableOpSoftmax/bias*
dtype0*
_output_shapes
:

�
Softmax/BiasAddBiasAddSoftmax/MatMulSoftmax/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:���������

]
Softmax/SoftmaxSoftmaxSoftmax/BiasAdd*
T0*'
_output_shapes
:���������

r
0TFOptimizer/iterations/Initializer/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
�
TFOptimizer/iterationsVarHandleOp*
dtype0	*
_output_shapes
: *'
shared_nameTFOptimizer/iterations*
shape: *
	container 
}
7TFOptimizer/iterations/IsInitialized/VarIsInitializedOpVarIsInitializedOpTFOptimizer/iterations*
_output_shapes
: 
�
TFOptimizer/iterations/AssignAssignVariableOpTFOptimizer/iterations0TFOptimizer/iterations/Initializer/initial_value*
dtype0	*)
_class
loc:@TFOptimizer/iterations
�
*TFOptimizer/iterations/Read/ReadVariableOpReadVariableOpTFOptimizer/iterations*
dtype0	*
_output_shapes
: *)
_class
loc:@TFOptimizer/iterations
�
Softmax_targetPlaceholder*
dtype0*0
_output_shapes
:������������������*%
shape:������������������
R
ConstConst*
valueB*  �?*
dtype0*
_output_shapes
:
�
Softmax_sample_weightsPlaceholderWithDefaultConst*
dtype0*#
_output_shapes
:���������*
shape:���������
\
loss/Softmax_loss/ConstConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
\
loss/Softmax_loss/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
o
loss/Softmax_loss/subSubloss/Softmax_loss/sub/xloss/Softmax_loss/Const*
T0*
_output_shapes
: 
�
'loss/Softmax_loss/clip_by_value/MinimumMinimumSoftmax/Softmaxloss/Softmax_loss/sub*
T0*'
_output_shapes
:���������

�
loss/Softmax_loss/clip_by_valueMaximum'loss/Softmax_loss/clip_by_value/Minimumloss/Softmax_loss/Const*
T0*'
_output_shapes
:���������

o
loss/Softmax_loss/LogLogloss/Softmax_loss/clip_by_value*
T0*'
_output_shapes
:���������

r
loss/Softmax_loss/Reshape/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
�
loss/Softmax_loss/ReshapeReshapeSoftmax_targetloss/Softmax_loss/Reshape/shape*
T0*
Tshape0*#
_output_shapes
:���������
v
loss/Softmax_loss/CastCastloss/Softmax_loss/Reshape*

DstT0	*#
_output_shapes
:���������*

SrcT0
r
!loss/Softmax_loss/Reshape_1/shapeConst*
valueB"����
   *
dtype0*
_output_shapes
:
�
loss/Softmax_loss/Reshape_1Reshapeloss/Softmax_loss/Log!loss/Softmax_loss/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:���������

�
;loss/Softmax_loss/SparseSoftmaxCrossEntropyWithLogits/ShapeShapeloss/Softmax_loss/Cast*
T0	*
out_type0*
_output_shapes
:
�
Yloss/Softmax_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsloss/Softmax_loss/Reshape_1loss/Softmax_loss/Cast*
T0*6
_output_shapes$
":���������:���������
*
Tlabels0	
k
(loss/Softmax_loss/Mean/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
�
loss/Softmax_loss/MeanMeanYloss/Softmax_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(loss/Softmax_loss/Mean/reduction_indices*
	keep_dims( *
T0*#
_output_shapes
:���������*

Tidx0
z
loss/Softmax_loss/mulMulloss/Softmax_loss/MeanSoftmax_sample_weights*
T0*#
_output_shapes
:���������
a
loss/Softmax_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
loss/Softmax_loss/NotEqualNotEqualSoftmax_sample_weightsloss/Softmax_loss/NotEqual/y*
T0*#
_output_shapes
:���������
y
loss/Softmax_loss/Cast_1Castloss/Softmax_loss/NotEqual*

DstT0*#
_output_shapes
:���������*

SrcT0

c
loss/Softmax_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
loss/Softmax_loss/Mean_1Meanloss/Softmax_loss/Cast_1loss/Softmax_loss/Const_1*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
�
loss/Softmax_loss/truedivRealDivloss/Softmax_loss/mulloss/Softmax_loss/Mean_1*
T0*#
_output_shapes
:���������
c
loss/Softmax_loss/Const_2Const*
valueB: *
dtype0*
_output_shapes
:
�
loss/Softmax_loss/Mean_2Meanloss/Softmax_loss/truedivloss/Softmax_loss/Const_2*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
O

loss/mul/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
V
loss/mulMul
loss/mul/xloss/Softmax_loss/Mean_2*
T0*
_output_shapes
: 
l
!metrics/acc/Max/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
metrics/acc/MaxMaxSoftmax_target!metrics/acc/Max/reduction_indices*
	keep_dims( *
T0*#
_output_shapes
:���������*

Tidx0
g
metrics/acc/ArgMax/dimensionConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
metrics/acc/ArgMaxArgMaxSoftmax/Softmaxmetrics/acc/ArgMax/dimension*
output_type0	*
T0*#
_output_shapes
:���������*

Tidx0
i
metrics/acc/CastCastmetrics/acc/ArgMax*

DstT0*#
_output_shapes
:���������*

SrcT0	
k
metrics/acc/EqualEqualmetrics/acc/Maxmetrics/acc/Cast*
T0*#
_output_shapes
:���������
j
metrics/acc/Cast_1Castmetrics/acc/Equal*

DstT0*#
_output_shapes
:���������*

SrcT0

[
metrics/acc/ConstConst*
valueB: *
dtype0*
_output_shapes
:
}
metrics/acc/MeanMeanmetrics/acc/Cast_1metrics/acc/Const*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
\
training/TFOptimizer/ConstConst*
value	B	 R*
dtype0	*
_output_shapes
: 
�
(training/TFOptimizer/AssignAddVariableOpAssignAddVariableOpTFOptimizer/iterationstraining/TFOptimizer/Const*
dtype0	
�
#training/TFOptimizer/ReadVariableOpReadVariableOpTFOptimizer/iterations)^training/TFOptimizer/AssignAddVariableOp*
dtype0	*
_output_shapes
: 
g
$training/TFOptimizer/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
m
(training/TFOptimizer/gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
#training/TFOptimizer/gradients/FillFill$training/TFOptimizer/gradients/Shape(training/TFOptimizer/gradients/grad_ys_0*
T0*
_output_shapes
: *

index_type0
�
0training/TFOptimizer/gradients/loss/mul_grad/MulMul#training/TFOptimizer/gradients/Fillloss/Softmax_loss/Mean_2*
T0*
_output_shapes
: 
�
2training/TFOptimizer/gradients/loss/mul_grad/Mul_1Mul#training/TFOptimizer/gradients/Fill
loss/mul/x*
T0*
_output_shapes
: 
�
=training/TFOptimizer/gradients/loss/mul_grad/tuple/group_depsNoOp1^training/TFOptimizer/gradients/loss/mul_grad/Mul3^training/TFOptimizer/gradients/loss/mul_grad/Mul_1
�
Etraining/TFOptimizer/gradients/loss/mul_grad/tuple/control_dependencyIdentity0training/TFOptimizer/gradients/loss/mul_grad/Mul>^training/TFOptimizer/gradients/loss/mul_grad/tuple/group_deps*
T0*
_output_shapes
: *C
_class9
75loc:@training/TFOptimizer/gradients/loss/mul_grad/Mul
�
Gtraining/TFOptimizer/gradients/loss/mul_grad/tuple/control_dependency_1Identity2training/TFOptimizer/gradients/loss/mul_grad/Mul_1>^training/TFOptimizer/gradients/loss/mul_grad/tuple/group_deps*
T0*
_output_shapes
: *E
_class;
97loc:@training/TFOptimizer/gradients/loss/mul_grad/Mul_1
�
Jtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
Dtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/ReshapeReshapeGtraining/TFOptimizer/gradients/loss/mul_grad/tuple/control_dependency_1Jtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
�
Btraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/ShapeShapeloss/Softmax_loss/truediv*
T0*
out_type0*
_output_shapes
:
�
Atraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/TileTileDtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/ReshapeBtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/Shape*
T0*#
_output_shapes
:���������*

Tmultiples0
�
Dtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/Shape_1Shapeloss/Softmax_loss/truediv*
T0*
out_type0*
_output_shapes
:
�
Dtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
�
Btraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
Atraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/ProdProdDtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/Shape_1Btraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/Const*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
�
Dtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
Ctraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/Prod_1ProdDtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/Shape_2Dtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/Const_1*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
�
Ftraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
Dtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/MaximumMaximumCtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/Prod_1Ftraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/Maximum/y*
T0*
_output_shapes
: 
�
Etraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/floordivFloorDivAtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/ProdDtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/Maximum*
T0*
_output_shapes
: 
�
Atraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/CastCastEtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/floordiv*

DstT0*
_output_shapes
: *

SrcT0
�
Dtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/truedivRealDivAtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/TileAtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/Cast*
T0*#
_output_shapes
:���������
�
Ctraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/ShapeShapeloss/Softmax_loss/mul*
T0*
out_type0*
_output_shapes
:
�
Etraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Straining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsCtraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/ShapeEtraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Etraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/RealDivRealDivDtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/truedivloss/Softmax_loss/Mean_1*
T0*#
_output_shapes
:���������
�
Atraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/SumSumEtraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/RealDivStraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
Etraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/ReshapeReshapeAtraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/SumCtraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
Atraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/NegNegloss/Softmax_loss/mul*
T0*#
_output_shapes
:���������
�
Gtraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/RealDiv_1RealDivAtraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/Negloss/Softmax_loss/Mean_1*
T0*#
_output_shapes
:���������
�
Gtraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/RealDiv_2RealDivGtraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/RealDiv_1loss/Softmax_loss/Mean_1*
T0*#
_output_shapes
:���������
�
Atraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/mulMulDtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_2_grad/truedivGtraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/RealDiv_2*
T0*#
_output_shapes
:���������
�
Ctraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/Sum_1SumAtraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/mulUtraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
Gtraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/Reshape_1ReshapeCtraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/Sum_1Etraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Ntraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/tuple/group_depsNoOpF^training/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/ReshapeH^training/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/Reshape_1
�
Vtraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/tuple/control_dependencyIdentityEtraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/ReshapeO^training/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/tuple/group_deps*
T0*#
_output_shapes
:���������*X
_classN
LJloc:@training/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/Reshape
�
Xtraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/tuple/control_dependency_1IdentityGtraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/Reshape_1O^training/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/tuple/group_deps*
T0*
_output_shapes
: *Z
_classP
NLloc:@training/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/Reshape_1
�
?training/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/ShapeShapeloss/Softmax_loss/Mean*
T0*
out_type0*
_output_shapes
:
�
Atraining/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/Shape_1ShapeSoftmax_sample_weights*
T0*
out_type0*
_output_shapes
:
�
Otraining/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs?training/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/ShapeAtraining/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
=training/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/MulMulVtraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/tuple/control_dependencySoftmax_sample_weights*
T0*#
_output_shapes
:���������
�
=training/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/SumSum=training/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/MulOtraining/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
Atraining/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/ReshapeReshape=training/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/Sum?training/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
?training/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/Mul_1Mulloss/Softmax_loss/MeanVtraining/TFOptimizer/gradients/loss/Softmax_loss/truediv_grad/tuple/control_dependency*
T0*#
_output_shapes
:���������
�
?training/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/Sum_1Sum?training/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/Mul_1Qtraining/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
Ctraining/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/Reshape_1Reshape?training/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/Sum_1Atraining/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:���������
�
Jtraining/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/tuple/group_depsNoOpB^training/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/ReshapeD^training/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/Reshape_1
�
Rtraining/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/tuple/control_dependencyIdentityAtraining/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/ReshapeK^training/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/tuple/group_deps*
T0*#
_output_shapes
:���������*T
_classJ
HFloc:@training/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/Reshape
�
Ttraining/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/tuple/control_dependency_1IdentityCtraining/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/Reshape_1K^training/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/tuple/group_deps*
T0*#
_output_shapes
:���������*V
_classL
JHloc:@training/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/Reshape_1
�
@training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/ShapeShapeYloss/Softmax_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
�
?training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: *S
_classI
GEloc:@training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Shape
�
>training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/addAdd(loss/Softmax_loss/Mean/reduction_indices?training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Size*
T0*
_output_shapes
: *S
_classI
GEloc:@training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Shape
�
>training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/modFloorMod>training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/add?training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Size*
T0*
_output_shapes
: *S
_classI
GEloc:@training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Shape
�
Btraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Shape_1Const*
valueB: *
dtype0*
_output_shapes
:*S
_classI
GEloc:@training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Shape
�
Ftraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: *S
_classI
GEloc:@training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Shape
�
Ftraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: *S
_classI
GEloc:@training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Shape
�
@training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/rangeRangeFtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/range/start?training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/SizeFtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/range/delta*
_output_shapes
:*

Tidx0*S
_classI
GEloc:@training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Shape
�
Etraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: *S
_classI
GEloc:@training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Shape
�
?training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/FillFillBtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Shape_1Etraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Fill/value*
T0*
_output_shapes
: *

index_type0*S
_classI
GEloc:@training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Shape
�
Htraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/DynamicStitchDynamicStitch@training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/range>training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/mod@training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Shape?training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Fill*
T0*
N*
_output_shapes
:*S
_classI
GEloc:@training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Shape
�
Dtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: *S
_classI
GEloc:@training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Shape
�
Btraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/MaximumMaximumHtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/DynamicStitchDtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Maximum/y*
T0*
_output_shapes
:*S
_classI
GEloc:@training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Shape
�
Ctraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/floordivFloorDiv@training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/ShapeBtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Maximum*
T0*
_output_shapes
:*S
_classI
GEloc:@training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Shape
�
Btraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/ReshapeReshapeRtraining/TFOptimizer/gradients/loss/Softmax_loss/mul_grad/tuple/control_dependencyHtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/DynamicStitch*
T0*
Tshape0*#
_output_shapes
:���������
�
?training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/TileTileBtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/ReshapeCtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/floordiv*
T0*#
_output_shapes
:���������*

Tmultiples0
�
Btraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Shape_2ShapeYloss/Softmax_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
�
Btraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Shape_3Shapeloss/Softmax_loss/Mean*
T0*
out_type0*
_output_shapes
:
�
@training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
?training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/ProdProdBtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Shape_2@training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Const*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
�
Btraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
Atraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Prod_1ProdBtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Shape_3Btraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Const_1*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
�
Ftraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
Dtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Maximum_1MaximumAtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Prod_1Ftraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Maximum_1/y*
T0*
_output_shapes
: 
�
Etraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/floordiv_1FloorDiv?training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/ProdDtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Maximum_1*
T0*
_output_shapes
: 
�
?training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/CastCastEtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/floordiv_1*

DstT0*
_output_shapes
: *

SrcT0
�
Btraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/truedivRealDiv?training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Tile?training/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/Cast*
T0*#
_output_shapes
:���������
�
)training/TFOptimizer/gradients/zeros_like	ZerosLike[loss/Softmax_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*'
_output_shapes
:���������

�
�training/TFOptimizer/gradients/loss/Softmax_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradient[loss/Softmax_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*�
message��Currently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*'
_output_shapes
:���������

�
�training/TFOptimizer/gradients/loss/Softmax_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
�training/TFOptimizer/gradients/loss/Softmax_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDimsBtraining/TFOptimizer/gradients/loss/Softmax_loss/Mean_grad/truediv�training/TFOptimizer/gradients/loss/Softmax_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*

Tdim0*'
_output_shapes
:���������
�
�training/TFOptimizer/gradients/loss/Softmax_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMul�training/TFOptimizer/gradients/loss/Softmax_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims�training/TFOptimizer/gradients/loss/Softmax_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*
T0*'
_output_shapes
:���������

�
Etraining/TFOptimizer/gradients/loss/Softmax_loss/Reshape_1_grad/ShapeShapeloss/Softmax_loss/Log*
T0*
out_type0*
_output_shapes
:
�
Gtraining/TFOptimizer/gradients/loss/Softmax_loss/Reshape_1_grad/ReshapeReshape�training/TFOptimizer/gradients/loss/Softmax_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulEtraining/TFOptimizer/gradients/loss/Softmax_loss/Reshape_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

�
Dtraining/TFOptimizer/gradients/loss/Softmax_loss/Log_grad/Reciprocal
Reciprocalloss/Softmax_loss/clip_by_valueH^training/TFOptimizer/gradients/loss/Softmax_loss/Reshape_1_grad/Reshape*
T0*'
_output_shapes
:���������

�
=training/TFOptimizer/gradients/loss/Softmax_loss/Log_grad/mulMulGtraining/TFOptimizer/gradients/loss/Softmax_loss/Reshape_1_grad/ReshapeDtraining/TFOptimizer/gradients/loss/Softmax_loss/Log_grad/Reciprocal*
T0*'
_output_shapes
:���������

�
Itraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/ShapeShape'loss/Softmax_loss/clip_by_value/Minimum*
T0*
out_type0*
_output_shapes
:
�
Ktraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Ktraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/Shape_2Shape=training/TFOptimizer/gradients/loss/Softmax_loss/Log_grad/mul*
T0*
out_type0*
_output_shapes
:
�
Otraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Itraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/zerosFillKtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/Shape_2Otraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/zeros/Const*
T0*'
_output_shapes
:���������
*

index_type0
�
Ptraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/GreaterEqualGreaterEqual'loss/Softmax_loss/clip_by_value/Minimumloss/Softmax_loss/Const*
T0*'
_output_shapes
:���������

�
Ytraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgsItraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/ShapeKtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Jtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/SelectSelectPtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/GreaterEqual=training/TFOptimizer/gradients/loss/Softmax_loss/Log_grad/mulItraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/zeros*
T0*'
_output_shapes
:���������

�
Ltraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/Select_1SelectPtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/GreaterEqualItraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/zeros=training/TFOptimizer/gradients/loss/Softmax_loss/Log_grad/mul*
T0*'
_output_shapes
:���������

�
Gtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/SumSumJtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/SelectYtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
Ktraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/ReshapeReshapeGtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/SumItraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

�
Itraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/Sum_1SumLtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/Select_1[training/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
Mtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/Reshape_1ReshapeItraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/Sum_1Ktraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Ttraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/tuple/group_depsNoOpL^training/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/ReshapeN^training/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/Reshape_1
�
\training/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/tuple/control_dependencyIdentityKtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/ReshapeU^training/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/tuple/group_deps*
T0*'
_output_shapes
:���������
*^
_classT
RPloc:@training/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/Reshape
�
^training/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/tuple/control_dependency_1IdentityMtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/Reshape_1U^training/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/tuple/group_deps*
T0*
_output_shapes
: *`
_classV
TRloc:@training/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/Reshape_1
�
Qtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/ShapeShapeSoftmax/Softmax*
T0*
out_type0*
_output_shapes
:
�
Straining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Straining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/Shape_2Shape\training/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
Wtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Qtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/zerosFillStraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/Shape_2Wtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/zeros/Const*
T0*'
_output_shapes
:���������
*

index_type0
�
Utraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/LessEqual	LessEqualSoftmax/Softmaxloss/Softmax_loss/sub*
T0*'
_output_shapes
:���������

�
atraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsQtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/ShapeStraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Rtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/SelectSelectUtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/LessEqual\training/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/tuple/control_dependencyQtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/zeros*
T0*'
_output_shapes
:���������

�
Ttraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/Select_1SelectUtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/LessEqualQtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/zeros\training/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������

�
Otraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/SumSumRtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/Selectatraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
Straining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/ReshapeReshapeOtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/SumQtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������

�
Qtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/Sum_1SumTtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/Select_1ctraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
Utraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/Reshape_1ReshapeQtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/Sum_1Straining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
\training/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/tuple/group_depsNoOpT^training/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/ReshapeV^training/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/Reshape_1
�
dtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/tuple/control_dependencyIdentityStraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/Reshape]^training/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/tuple/group_deps*
T0*'
_output_shapes
:���������
*f
_class\
ZXloc:@training/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/Reshape
�
ftraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/tuple/control_dependency_1IdentityUtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/Reshape_1]^training/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/tuple/group_deps*
T0*
_output_shapes
: *h
_class^
\Zloc:@training/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/Reshape_1
�
7training/TFOptimizer/gradients/Softmax/Softmax_grad/mulMuldtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/tuple/control_dependencySoftmax/Softmax*
T0*'
_output_shapes
:���������

�
Itraining/TFOptimizer/gradients/Softmax/Softmax_grad/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
7training/TFOptimizer/gradients/Softmax/Softmax_grad/SumSum7training/TFOptimizer/gradients/Softmax/Softmax_grad/mulItraining/TFOptimizer/gradients/Softmax/Softmax_grad/Sum/reduction_indices*
	keep_dims( *
T0*#
_output_shapes
:���������*

Tidx0
�
Atraining/TFOptimizer/gradients/Softmax/Softmax_grad/Reshape/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:
�
;training/TFOptimizer/gradients/Softmax/Softmax_grad/ReshapeReshape7training/TFOptimizer/gradients/Softmax/Softmax_grad/SumAtraining/TFOptimizer/gradients/Softmax/Softmax_grad/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:���������
�
7training/TFOptimizer/gradients/Softmax/Softmax_grad/subSubdtraining/TFOptimizer/gradients/loss/Softmax_loss/clip_by_value/Minimum_grad/tuple/control_dependency;training/TFOptimizer/gradients/Softmax/Softmax_grad/Reshape*
T0*'
_output_shapes
:���������

�
9training/TFOptimizer/gradients/Softmax/Softmax_grad/mul_1Mul7training/TFOptimizer/gradients/Softmax/Softmax_grad/subSoftmax/Softmax*
T0*'
_output_shapes
:���������

�
?training/TFOptimizer/gradients/Softmax/BiasAdd_grad/BiasAddGradBiasAddGrad9training/TFOptimizer/gradients/Softmax/Softmax_grad/mul_1*
T0*
data_formatNHWC*
_output_shapes
:

�
Dtraining/TFOptimizer/gradients/Softmax/BiasAdd_grad/tuple/group_depsNoOp@^training/TFOptimizer/gradients/Softmax/BiasAdd_grad/BiasAddGrad:^training/TFOptimizer/gradients/Softmax/Softmax_grad/mul_1
�
Ltraining/TFOptimizer/gradients/Softmax/BiasAdd_grad/tuple/control_dependencyIdentity9training/TFOptimizer/gradients/Softmax/Softmax_grad/mul_1E^training/TFOptimizer/gradients/Softmax/BiasAdd_grad/tuple/group_deps*
T0*'
_output_shapes
:���������
*L
_classB
@>loc:@training/TFOptimizer/gradients/Softmax/Softmax_grad/mul_1
�
Ntraining/TFOptimizer/gradients/Softmax/BiasAdd_grad/tuple/control_dependency_1Identity?training/TFOptimizer/gradients/Softmax/BiasAdd_grad/BiasAddGradE^training/TFOptimizer/gradients/Softmax/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:
*R
_classH
FDloc:@training/TFOptimizer/gradients/Softmax/BiasAdd_grad/BiasAddGrad
�
9training/TFOptimizer/gradients/Softmax/MatMul_grad/MatMulMatMulLtraining/TFOptimizer/gradients/Softmax/BiasAdd_grad/tuple/control_dependencySoftmax/MatMul/ReadVariableOp*
T0*
transpose_b(*
transpose_a( *(
_output_shapes
:����������

�
;training/TFOptimizer/gradients/Softmax/MatMul_grad/MatMul_1MatMulflatten/ReshapeLtraining/TFOptimizer/gradients/Softmax/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_b( *
transpose_a(*
_output_shapes
:	�


�
Ctraining/TFOptimizer/gradients/Softmax/MatMul_grad/tuple/group_depsNoOp:^training/TFOptimizer/gradients/Softmax/MatMul_grad/MatMul<^training/TFOptimizer/gradients/Softmax/MatMul_grad/MatMul_1
�
Ktraining/TFOptimizer/gradients/Softmax/MatMul_grad/tuple/control_dependencyIdentity9training/TFOptimizer/gradients/Softmax/MatMul_grad/MatMulD^training/TFOptimizer/gradients/Softmax/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:����������
*L
_classB
@>loc:@training/TFOptimizer/gradients/Softmax/MatMul_grad/MatMul
�
Mtraining/TFOptimizer/gradients/Softmax/MatMul_grad/tuple/control_dependency_1Identity;training/TFOptimizer/gradients/Softmax/MatMul_grad/MatMul_1D^training/TFOptimizer/gradients/Softmax/MatMul_grad/tuple/group_deps*
T0*
_output_shapes
:	�

*N
_classD
B@loc:@training/TFOptimizer/gradients/Softmax/MatMul_grad/MatMul_1
�
9training/TFOptimizer/gradients/flatten/Reshape_grad/ShapeShape
Conv1/Relu*
T0*
out_type0*
_output_shapes
:
�
;training/TFOptimizer/gradients/flatten/Reshape_grad/ReshapeReshapeKtraining/TFOptimizer/gradients/Softmax/MatMul_grad/tuple/control_dependency9training/TFOptimizer/gradients/flatten/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������
�
7training/TFOptimizer/gradients/Conv1/Relu_grad/ReluGradReluGrad;training/TFOptimizer/gradients/flatten/Reshape_grad/Reshape
Conv1/Relu*
T0*/
_output_shapes
:���������
�
=training/TFOptimizer/gradients/Conv1/BiasAdd_grad/BiasAddGradBiasAddGrad7training/TFOptimizer/gradients/Conv1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
�
Btraining/TFOptimizer/gradients/Conv1/BiasAdd_grad/tuple/group_depsNoOp>^training/TFOptimizer/gradients/Conv1/BiasAdd_grad/BiasAddGrad8^training/TFOptimizer/gradients/Conv1/Relu_grad/ReluGrad
�
Jtraining/TFOptimizer/gradients/Conv1/BiasAdd_grad/tuple/control_dependencyIdentity7training/TFOptimizer/gradients/Conv1/Relu_grad/ReluGradC^training/TFOptimizer/gradients/Conv1/BiasAdd_grad/tuple/group_deps*
T0*/
_output_shapes
:���������*J
_class@
><loc:@training/TFOptimizer/gradients/Conv1/Relu_grad/ReluGrad
�
Ltraining/TFOptimizer/gradients/Conv1/BiasAdd_grad/tuple/control_dependency_1Identity=training/TFOptimizer/gradients/Conv1/BiasAdd_grad/BiasAddGradC^training/TFOptimizer/gradients/Conv1/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:*P
_classF
DBloc:@training/TFOptimizer/gradients/Conv1/BiasAdd_grad/BiasAddGrad
�
7training/TFOptimizer/gradients/Conv1/Conv2D_grad/ShapeNShapeNConv1_inputConv1/Conv2D/ReadVariableOp*
T0*
out_type0*
N* 
_output_shapes
::
�
Dtraining/TFOptimizer/gradients/Conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput7training/TFOptimizer/gradients/Conv1/Conv2D_grad/ShapeNConv1/Conv2D/ReadVariableOpJtraining/TFOptimizer/gradients/Conv1/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*/
_output_shapes
:���������*
use_cudnn_on_gpu(
�
Etraining/TFOptimizer/gradients/Conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterConv1_input9training/TFOptimizer/gradients/Conv1/Conv2D_grad/ShapeN:1Jtraining/TFOptimizer/gradients/Conv1/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*&
_output_shapes
:*
use_cudnn_on_gpu(
�
Atraining/TFOptimizer/gradients/Conv1/Conv2D_grad/tuple/group_depsNoOpF^training/TFOptimizer/gradients/Conv1/Conv2D_grad/Conv2DBackpropFilterE^training/TFOptimizer/gradients/Conv1/Conv2D_grad/Conv2DBackpropInput
�
Itraining/TFOptimizer/gradients/Conv1/Conv2D_grad/tuple/control_dependencyIdentityDtraining/TFOptimizer/gradients/Conv1/Conv2D_grad/Conv2DBackpropInputB^training/TFOptimizer/gradients/Conv1/Conv2D_grad/tuple/group_deps*
T0*/
_output_shapes
:���������*W
_classM
KIloc:@training/TFOptimizer/gradients/Conv1/Conv2D_grad/Conv2DBackpropInput
�
Ktraining/TFOptimizer/gradients/Conv1/Conv2D_grad/tuple/control_dependency_1IdentityEtraining/TFOptimizer/gradients/Conv1/Conv2D_grad/Conv2DBackpropFilterB^training/TFOptimizer/gradients/Conv1/Conv2D_grad/tuple/group_deps*
T0*&
_output_shapes
:*X
_classN
LJloc:@training/TFOptimizer/gradients/Conv1/Conv2D_grad/Conv2DBackpropFilter
o
(training/TFOptimizer/Read/ReadVariableOpReadVariableOp
Conv1/bias*
dtype0*
_output_shapes
:
x
training/TFOptimizer/IdentityIdentity(training/TFOptimizer/Read/ReadVariableOp*
T0*
_output_shapes
:
�
.training/TFOptimizer/beta1_power/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: *;
_class1
/-loc:@training/TFOptimizer/Read/ReadVariableOp
�
 training/TFOptimizer/beta1_power
VariableV2*
shared_name *;
_class1
/-loc:@training/TFOptimizer/Read/ReadVariableOp*
_output_shapes
: *
dtype0*
shape: *
	container 
�
'training/TFOptimizer/beta1_power/AssignAssign training/TFOptimizer/beta1_power.training/TFOptimizer/beta1_power/initial_value*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*;
_class1
/-loc:@training/TFOptimizer/Read/ReadVariableOp
�
%training/TFOptimizer/beta1_power/readIdentity training/TFOptimizer/beta1_power*
T0*
_output_shapes
: *;
_class1
/-loc:@training/TFOptimizer/Read/ReadVariableOp
q
*training/TFOptimizer/Read_1/ReadVariableOpReadVariableOp
Conv1/bias*
dtype0*
_output_shapes
:
|
training/TFOptimizer/Identity_1Identity*training/TFOptimizer/Read_1/ReadVariableOp*
T0*
_output_shapes
:
�
.training/TFOptimizer/beta2_power/initial_valueConst*
valueB
 *w�?*
dtype0*
_output_shapes
: *=
_class3
1/loc:@training/TFOptimizer/Read_1/ReadVariableOp
�
 training/TFOptimizer/beta2_power
VariableV2*
shared_name *=
_class3
1/loc:@training/TFOptimizer/Read_1/ReadVariableOp*
_output_shapes
: *
dtype0*
shape: *
	container 
�
'training/TFOptimizer/beta2_power/AssignAssign training/TFOptimizer/beta2_power.training/TFOptimizer/beta2_power/initial_value*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*=
_class3
1/loc:@training/TFOptimizer/Read_1/ReadVariableOp
�
%training/TFOptimizer/beta2_power/readIdentity training/TFOptimizer/beta2_power*
T0*
_output_shapes
: *=
_class3
1/loc:@training/TFOptimizer/Read_1/ReadVariableOp
�
:training/TFOptimizer/Conv1/kernel/Adam/Read/ReadVariableOpReadVariableOpConv1/kernel*
dtype0*&
_output_shapes
:
�
/training/TFOptimizer/Conv1/kernel/Adam/IdentityIdentity:training/TFOptimizer/Conv1/kernel/Adam/Read/ReadVariableOp*
T0*&
_output_shapes
:
�
#Conv1/kernel/Adam/Initializer/zerosConst*%
valueB*    *
dtype0*&
_output_shapes
:*M
_classC
A?loc:@training/TFOptimizer/Conv1/kernel/Adam/Read/ReadVariableOp
�
Conv1/kernel/AdamVarHandleOp*"
shared_nameConv1/kernel/Adam*M
_classC
A?loc:@training/TFOptimizer/Conv1/kernel/Adam/Read/ReadVariableOp*
_output_shapes
: *
dtype0*
	container *
shape:
�
2Conv1/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpConv1/kernel/Adam*
_output_shapes
: *M
_classC
A?loc:@training/TFOptimizer/Conv1/kernel/Adam/Read/ReadVariableOp
�
Conv1/kernel/Adam/AssignAssignVariableOpConv1/kernel/Adam#Conv1/kernel/Adam/Initializer/zeros*
dtype0*M
_classC
A?loc:@training/TFOptimizer/Conv1/kernel/Adam/Read/ReadVariableOp
�
%Conv1/kernel/Adam/Read/ReadVariableOpReadVariableOpConv1/kernel/Adam*
dtype0*&
_output_shapes
:*M
_classC
A?loc:@training/TFOptimizer/Conv1/kernel/Adam/Read/ReadVariableOp
�
<training/TFOptimizer/Conv1/kernel/Adam_1/Read/ReadVariableOpReadVariableOpConv1/kernel*
dtype0*&
_output_shapes
:
�
1training/TFOptimizer/Conv1/kernel/Adam_1/IdentityIdentity<training/TFOptimizer/Conv1/kernel/Adam_1/Read/ReadVariableOp*
T0*&
_output_shapes
:
�
%Conv1/kernel/Adam_1/Initializer/zerosConst*%
valueB*    *
dtype0*&
_output_shapes
:*O
_classE
CAloc:@training/TFOptimizer/Conv1/kernel/Adam_1/Read/ReadVariableOp
�
Conv1/kernel/Adam_1VarHandleOp*$
shared_nameConv1/kernel/Adam_1*O
_classE
CAloc:@training/TFOptimizer/Conv1/kernel/Adam_1/Read/ReadVariableOp*
_output_shapes
: *
dtype0*
	container *
shape:
�
4Conv1/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpConv1/kernel/Adam_1*
_output_shapes
: *O
_classE
CAloc:@training/TFOptimizer/Conv1/kernel/Adam_1/Read/ReadVariableOp
�
Conv1/kernel/Adam_1/AssignAssignVariableOpConv1/kernel/Adam_1%Conv1/kernel/Adam_1/Initializer/zeros*
dtype0*O
_classE
CAloc:@training/TFOptimizer/Conv1/kernel/Adam_1/Read/ReadVariableOp
�
'Conv1/kernel/Adam_1/Read/ReadVariableOpReadVariableOpConv1/kernel/Adam_1*
dtype0*&
_output_shapes
:*O
_classE
CAloc:@training/TFOptimizer/Conv1/kernel/Adam_1/Read/ReadVariableOp

8training/TFOptimizer/Conv1/bias/Adam/Read/ReadVariableOpReadVariableOp
Conv1/bias*
dtype0*
_output_shapes
:
�
-training/TFOptimizer/Conv1/bias/Adam/IdentityIdentity8training/TFOptimizer/Conv1/bias/Adam/Read/ReadVariableOp*
T0*
_output_shapes
:
�
!Conv1/bias/Adam/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes
:*K
_classA
?=loc:@training/TFOptimizer/Conv1/bias/Adam/Read/ReadVariableOp
�
Conv1/bias/AdamVarHandleOp* 
shared_nameConv1/bias/Adam*K
_classA
?=loc:@training/TFOptimizer/Conv1/bias/Adam/Read/ReadVariableOp*
_output_shapes
: *
dtype0*
	container *
shape:
�
0Conv1/bias/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpConv1/bias/Adam*
_output_shapes
: *K
_classA
?=loc:@training/TFOptimizer/Conv1/bias/Adam/Read/ReadVariableOp
�
Conv1/bias/Adam/AssignAssignVariableOpConv1/bias/Adam!Conv1/bias/Adam/Initializer/zeros*
dtype0*K
_classA
?=loc:@training/TFOptimizer/Conv1/bias/Adam/Read/ReadVariableOp
�
#Conv1/bias/Adam/Read/ReadVariableOpReadVariableOpConv1/bias/Adam*
dtype0*
_output_shapes
:*K
_classA
?=loc:@training/TFOptimizer/Conv1/bias/Adam/Read/ReadVariableOp
�
:training/TFOptimizer/Conv1/bias/Adam_1/Read/ReadVariableOpReadVariableOp
Conv1/bias*
dtype0*
_output_shapes
:
�
/training/TFOptimizer/Conv1/bias/Adam_1/IdentityIdentity:training/TFOptimizer/Conv1/bias/Adam_1/Read/ReadVariableOp*
T0*
_output_shapes
:
�
#Conv1/bias/Adam_1/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes
:*M
_classC
A?loc:@training/TFOptimizer/Conv1/bias/Adam_1/Read/ReadVariableOp
�
Conv1/bias/Adam_1VarHandleOp*"
shared_nameConv1/bias/Adam_1*M
_classC
A?loc:@training/TFOptimizer/Conv1/bias/Adam_1/Read/ReadVariableOp*
_output_shapes
: *
dtype0*
	container *
shape:
�
2Conv1/bias/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpConv1/bias/Adam_1*
_output_shapes
: *M
_classC
A?loc:@training/TFOptimizer/Conv1/bias/Adam_1/Read/ReadVariableOp
�
Conv1/bias/Adam_1/AssignAssignVariableOpConv1/bias/Adam_1#Conv1/bias/Adam_1/Initializer/zeros*
dtype0*M
_classC
A?loc:@training/TFOptimizer/Conv1/bias/Adam_1/Read/ReadVariableOp
�
%Conv1/bias/Adam_1/Read/ReadVariableOpReadVariableOpConv1/bias/Adam_1*
dtype0*
_output_shapes
:*M
_classC
A?loc:@training/TFOptimizer/Conv1/bias/Adam_1/Read/ReadVariableOp
�
<training/TFOptimizer/Softmax/kernel/Adam/Read/ReadVariableOpReadVariableOpSoftmax/kernel*
dtype0*
_output_shapes
:	�


�
1training/TFOptimizer/Softmax/kernel/Adam/IdentityIdentity<training/TFOptimizer/Softmax/kernel/Adam/Read/ReadVariableOp*
T0*
_output_shapes
:	�


�
5Softmax/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"H  
   *
dtype0*
_output_shapes
:*O
_classE
CAloc:@training/TFOptimizer/Softmax/kernel/Adam/Read/ReadVariableOp
�
+Softmax/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *O
_classE
CAloc:@training/TFOptimizer/Softmax/kernel/Adam/Read/ReadVariableOp
�
%Softmax/kernel/Adam/Initializer/zerosFill5Softmax/kernel/Adam/Initializer/zeros/shape_as_tensor+Softmax/kernel/Adam/Initializer/zeros/Const*
T0*
_output_shapes
:	�

*

index_type0*O
_classE
CAloc:@training/TFOptimizer/Softmax/kernel/Adam/Read/ReadVariableOp
�
Softmax/kernel/AdamVarHandleOp*$
shared_nameSoftmax/kernel/Adam*O
_classE
CAloc:@training/TFOptimizer/Softmax/kernel/Adam/Read/ReadVariableOp*
_output_shapes
: *
dtype0*
	container *
shape:	�


�
4Softmax/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpSoftmax/kernel/Adam*
_output_shapes
: *O
_classE
CAloc:@training/TFOptimizer/Softmax/kernel/Adam/Read/ReadVariableOp
�
Softmax/kernel/Adam/AssignAssignVariableOpSoftmax/kernel/Adam%Softmax/kernel/Adam/Initializer/zeros*
dtype0*O
_classE
CAloc:@training/TFOptimizer/Softmax/kernel/Adam/Read/ReadVariableOp
�
'Softmax/kernel/Adam/Read/ReadVariableOpReadVariableOpSoftmax/kernel/Adam*
dtype0*
_output_shapes
:	�

*O
_classE
CAloc:@training/TFOptimizer/Softmax/kernel/Adam/Read/ReadVariableOp
�
>training/TFOptimizer/Softmax/kernel/Adam_1/Read/ReadVariableOpReadVariableOpSoftmax/kernel*
dtype0*
_output_shapes
:	�


�
3training/TFOptimizer/Softmax/kernel/Adam_1/IdentityIdentity>training/TFOptimizer/Softmax/kernel/Adam_1/Read/ReadVariableOp*
T0*
_output_shapes
:	�


�
7Softmax/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"H  
   *
dtype0*
_output_shapes
:*Q
_classG
ECloc:@training/TFOptimizer/Softmax/kernel/Adam_1/Read/ReadVariableOp
�
-Softmax/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *Q
_classG
ECloc:@training/TFOptimizer/Softmax/kernel/Adam_1/Read/ReadVariableOp
�
'Softmax/kernel/Adam_1/Initializer/zerosFill7Softmax/kernel/Adam_1/Initializer/zeros/shape_as_tensor-Softmax/kernel/Adam_1/Initializer/zeros/Const*
T0*
_output_shapes
:	�

*

index_type0*Q
_classG
ECloc:@training/TFOptimizer/Softmax/kernel/Adam_1/Read/ReadVariableOp
�
Softmax/kernel/Adam_1VarHandleOp*&
shared_nameSoftmax/kernel/Adam_1*Q
_classG
ECloc:@training/TFOptimizer/Softmax/kernel/Adam_1/Read/ReadVariableOp*
_output_shapes
: *
dtype0*
	container *
shape:	�


�
6Softmax/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpSoftmax/kernel/Adam_1*
_output_shapes
: *Q
_classG
ECloc:@training/TFOptimizer/Softmax/kernel/Adam_1/Read/ReadVariableOp
�
Softmax/kernel/Adam_1/AssignAssignVariableOpSoftmax/kernel/Adam_1'Softmax/kernel/Adam_1/Initializer/zeros*
dtype0*Q
_classG
ECloc:@training/TFOptimizer/Softmax/kernel/Adam_1/Read/ReadVariableOp
�
)Softmax/kernel/Adam_1/Read/ReadVariableOpReadVariableOpSoftmax/kernel/Adam_1*
dtype0*
_output_shapes
:	�

*Q
_classG
ECloc:@training/TFOptimizer/Softmax/kernel/Adam_1/Read/ReadVariableOp
�
:training/TFOptimizer/Softmax/bias/Adam/Read/ReadVariableOpReadVariableOpSoftmax/bias*
dtype0*
_output_shapes
:

�
/training/TFOptimizer/Softmax/bias/Adam/IdentityIdentity:training/TFOptimizer/Softmax/bias/Adam/Read/ReadVariableOp*
T0*
_output_shapes
:

�
#Softmax/bias/Adam/Initializer/zerosConst*
valueB
*    *
dtype0*
_output_shapes
:
*M
_classC
A?loc:@training/TFOptimizer/Softmax/bias/Adam/Read/ReadVariableOp
�
Softmax/bias/AdamVarHandleOp*"
shared_nameSoftmax/bias/Adam*M
_classC
A?loc:@training/TFOptimizer/Softmax/bias/Adam/Read/ReadVariableOp*
_output_shapes
: *
dtype0*
	container *
shape:

�
2Softmax/bias/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpSoftmax/bias/Adam*
_output_shapes
: *M
_classC
A?loc:@training/TFOptimizer/Softmax/bias/Adam/Read/ReadVariableOp
�
Softmax/bias/Adam/AssignAssignVariableOpSoftmax/bias/Adam#Softmax/bias/Adam/Initializer/zeros*
dtype0*M
_classC
A?loc:@training/TFOptimizer/Softmax/bias/Adam/Read/ReadVariableOp
�
%Softmax/bias/Adam/Read/ReadVariableOpReadVariableOpSoftmax/bias/Adam*
dtype0*
_output_shapes
:
*M
_classC
A?loc:@training/TFOptimizer/Softmax/bias/Adam/Read/ReadVariableOp
�
<training/TFOptimizer/Softmax/bias/Adam_1/Read/ReadVariableOpReadVariableOpSoftmax/bias*
dtype0*
_output_shapes
:

�
1training/TFOptimizer/Softmax/bias/Adam_1/IdentityIdentity<training/TFOptimizer/Softmax/bias/Adam_1/Read/ReadVariableOp*
T0*
_output_shapes
:

�
%Softmax/bias/Adam_1/Initializer/zerosConst*
valueB
*    *
dtype0*
_output_shapes
:
*O
_classE
CAloc:@training/TFOptimizer/Softmax/bias/Adam_1/Read/ReadVariableOp
�
Softmax/bias/Adam_1VarHandleOp*$
shared_nameSoftmax/bias/Adam_1*O
_classE
CAloc:@training/TFOptimizer/Softmax/bias/Adam_1/Read/ReadVariableOp*
_output_shapes
: *
dtype0*
	container *
shape:

�
4Softmax/bias/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpSoftmax/bias/Adam_1*
_output_shapes
: *O
_classE
CAloc:@training/TFOptimizer/Softmax/bias/Adam_1/Read/ReadVariableOp
�
Softmax/bias/Adam_1/AssignAssignVariableOpSoftmax/bias/Adam_1%Softmax/bias/Adam_1/Initializer/zeros*
dtype0*O
_classE
CAloc:@training/TFOptimizer/Softmax/bias/Adam_1/Read/ReadVariableOp
�
'Softmax/bias/Adam_1/Read/ReadVariableOpReadVariableOpSoftmax/bias/Adam_1*
dtype0*
_output_shapes
:
*O
_classE
CAloc:@training/TFOptimizer/Softmax/bias/Adam_1/Read/ReadVariableOp
l
'training/TFOptimizer/Adam/learning_rateConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
d
training/TFOptimizer/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
d
training/TFOptimizer/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
f
!training/TFOptimizer/Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
Atraining/TFOptimizer/Adam/update_Conv1/kernel/Read/ReadVariableOpReadVariableOpConv1/kernel*
dtype0*&
_output_shapes
:
�
6training/TFOptimizer/Adam/update_Conv1/kernel/IdentityIdentityAtraining/TFOptimizer/Adam/update_Conv1/kernel/Read/ReadVariableOp*
T0*&
_output_shapes
:
�
?training/TFOptimizer/Adam/update_Conv1/kernel/ResourceApplyAdamResourceApplyAdamConv1/kernelConv1/kernel/AdamConv1/kernel/Adam_1%training/TFOptimizer/beta1_power/read%training/TFOptimizer/beta2_power/read'training/TFOptimizer/Adam/learning_ratetraining/TFOptimizer/Adam/beta1training/TFOptimizer/Adam/beta2!training/TFOptimizer/Adam/epsilonKtraining/TFOptimizer/gradients/Conv1/Conv2D_grad/tuple/control_dependency_1*
T0*
use_locking( *T
_classJ
HFloc:@training/TFOptimizer/Adam/update_Conv1/kernel/Read/ReadVariableOp*
use_nesterov( 
�
?training/TFOptimizer/Adam/update_Conv1/bias/Read/ReadVariableOpReadVariableOp
Conv1/bias*
dtype0*
_output_shapes
:
�
4training/TFOptimizer/Adam/update_Conv1/bias/IdentityIdentity?training/TFOptimizer/Adam/update_Conv1/bias/Read/ReadVariableOp*
T0*
_output_shapes
:
�
=training/TFOptimizer/Adam/update_Conv1/bias/ResourceApplyAdamResourceApplyAdam
Conv1/biasConv1/bias/AdamConv1/bias/Adam_1%training/TFOptimizer/beta1_power/read%training/TFOptimizer/beta2_power/read'training/TFOptimizer/Adam/learning_ratetraining/TFOptimizer/Adam/beta1training/TFOptimizer/Adam/beta2!training/TFOptimizer/Adam/epsilonLtraining/TFOptimizer/gradients/Conv1/BiasAdd_grad/tuple/control_dependency_1*
T0*
use_locking( *R
_classH
FDloc:@training/TFOptimizer/Adam/update_Conv1/bias/Read/ReadVariableOp*
use_nesterov( 
�
Ctraining/TFOptimizer/Adam/update_Softmax/kernel/Read/ReadVariableOpReadVariableOpSoftmax/kernel*
dtype0*
_output_shapes
:	�


�
8training/TFOptimizer/Adam/update_Softmax/kernel/IdentityIdentityCtraining/TFOptimizer/Adam/update_Softmax/kernel/Read/ReadVariableOp*
T0*
_output_shapes
:	�


�
Atraining/TFOptimizer/Adam/update_Softmax/kernel/ResourceApplyAdamResourceApplyAdamSoftmax/kernelSoftmax/kernel/AdamSoftmax/kernel/Adam_1%training/TFOptimizer/beta1_power/read%training/TFOptimizer/beta2_power/read'training/TFOptimizer/Adam/learning_ratetraining/TFOptimizer/Adam/beta1training/TFOptimizer/Adam/beta2!training/TFOptimizer/Adam/epsilonMtraining/TFOptimizer/gradients/Softmax/MatMul_grad/tuple/control_dependency_1*
T0*
use_locking( *V
_classL
JHloc:@training/TFOptimizer/Adam/update_Softmax/kernel/Read/ReadVariableOp*
use_nesterov( 
�
Atraining/TFOptimizer/Adam/update_Softmax/bias/Read/ReadVariableOpReadVariableOpSoftmax/bias*
dtype0*
_output_shapes
:

�
6training/TFOptimizer/Adam/update_Softmax/bias/IdentityIdentityAtraining/TFOptimizer/Adam/update_Softmax/bias/Read/ReadVariableOp*
T0*
_output_shapes
:

�
?training/TFOptimizer/Adam/update_Softmax/bias/ResourceApplyAdamResourceApplyAdamSoftmax/biasSoftmax/bias/AdamSoftmax/bias/Adam_1%training/TFOptimizer/beta1_power/read%training/TFOptimizer/beta2_power/read'training/TFOptimizer/Adam/learning_ratetraining/TFOptimizer/Adam/beta1training/TFOptimizer/Adam/beta2!training/TFOptimizer/Adam/epsilonNtraining/TFOptimizer/gradients/Softmax/BiasAdd_grad/tuple/control_dependency_1*
T0*
use_locking( *T
_classJ
HFloc:@training/TFOptimizer/Adam/update_Softmax/bias/Read/ReadVariableOp*
use_nesterov( 
�
training/TFOptimizer/Adam/mulMul%training/TFOptimizer/beta1_power/readtraining/TFOptimizer/Adam/beta1>^training/TFOptimizer/Adam/update_Conv1/bias/ResourceApplyAdam@^training/TFOptimizer/Adam/update_Conv1/kernel/ResourceApplyAdam@^training/TFOptimizer/Adam/update_Softmax/bias/ResourceApplyAdamB^training/TFOptimizer/Adam/update_Softmax/kernel/ResourceApplyAdam*
T0*
_output_shapes
: *;
_class1
/-loc:@training/TFOptimizer/Read/ReadVariableOp
�
 training/TFOptimizer/Adam/AssignAssign training/TFOptimizer/beta1_powertraining/TFOptimizer/Adam/mul*
T0*
_output_shapes
: *
use_locking( *
validate_shape(*;
_class1
/-loc:@training/TFOptimizer/Read/ReadVariableOp
�
training/TFOptimizer/Adam/mul_1Mul%training/TFOptimizer/beta2_power/readtraining/TFOptimizer/Adam/beta2>^training/TFOptimizer/Adam/update_Conv1/bias/ResourceApplyAdam@^training/TFOptimizer/Adam/update_Conv1/kernel/ResourceApplyAdam@^training/TFOptimizer/Adam/update_Softmax/bias/ResourceApplyAdamB^training/TFOptimizer/Adam/update_Softmax/kernel/ResourceApplyAdam*
T0*
_output_shapes
: *;
_class1
/-loc:@training/TFOptimizer/Read/ReadVariableOp
�
"training/TFOptimizer/Adam/Assign_1Assign training/TFOptimizer/beta2_powertraining/TFOptimizer/Adam/mul_1*
T0*
_output_shapes
: *
use_locking( *
validate_shape(*l
_classb
`-loc:@training/TFOptimizer/Read/ReadVariableOp/loc:@training/TFOptimizer/Read_1/ReadVariableOp
�
 training/TFOptimizer/Adam/updateNoOp!^training/TFOptimizer/Adam/Assign#^training/TFOptimizer/Adam/Assign_1>^training/TFOptimizer/Adam/update_Conv1/bias/ResourceApplyAdam@^training/TFOptimizer/Adam/update_Conv1/kernel/ResourceApplyAdam@^training/TFOptimizer/Adam/update_Softmax/bias/ResourceApplyAdamB^training/TFOptimizer/Adam/update_Softmax/kernel/ResourceApplyAdam
�
-training/TFOptimizer/Adam/Read/ReadVariableOpReadVariableOpTFOptimizer/iterations!^training/TFOptimizer/Adam/update*
dtype0	*
_output_shapes
: 
~
"training/TFOptimizer/Adam/IdentityIdentity-training/TFOptimizer/Adam/Read/ReadVariableOp*
T0	*
_output_shapes
: 
�
training/TFOptimizer/Adam/ConstConst!^training/TFOptimizer/Adam/update*
value	B	 R*
dtype0	*
_output_shapes
: *@
_class6
42loc:@training/TFOptimizer/Adam/Read/ReadVariableOp
�
training/TFOptimizer/AdamAssignAddVariableOpTFOptimizer/iterationstraining/TFOptimizer/Adam/Const*
dtype0	*@
_class6
42loc:@training/TFOptimizer/Adam/Read/ReadVariableOp
{
training/group_depsNoOp	^loss/mul^metrics/acc/Mean^training/TFOptimizer/Adam$^training/TFOptimizer/ReadVariableOp
N
VarIsInitializedOpVarIsInitializedOpConv1/kernel*
_output_shapes
: 
N
VarIsInitializedOp_1VarIsInitializedOp
Conv1/bias*
_output_shapes
: 
R
VarIsInitializedOp_2VarIsInitializedOpSoftmax/kernel*
_output_shapes
: 
P
VarIsInitializedOp_3VarIsInitializedOpSoftmax/bias*
_output_shapes
: 
Z
VarIsInitializedOp_4VarIsInitializedOpTFOptimizer/iterations*
_output_shapes
: 
�
IsVariableInitializedIsVariableInitialized training/TFOptimizer/beta1_power*
dtype0*
_output_shapes
: *;
_class1
/-loc:@training/TFOptimizer/Read/ReadVariableOp
�
IsVariableInitialized_1IsVariableInitialized training/TFOptimizer/beta2_power*
dtype0*
_output_shapes
: *=
_class3
1/loc:@training/TFOptimizer/Read_1/ReadVariableOp
U
VarIsInitializedOp_5VarIsInitializedOpConv1/kernel/Adam*
_output_shapes
: 
W
VarIsInitializedOp_6VarIsInitializedOpConv1/kernel/Adam_1*
_output_shapes
: 
S
VarIsInitializedOp_7VarIsInitializedOpConv1/bias/Adam*
_output_shapes
: 
U
VarIsInitializedOp_8VarIsInitializedOpConv1/bias/Adam_1*
_output_shapes
: 
W
VarIsInitializedOp_9VarIsInitializedOpSoftmax/kernel/Adam*
_output_shapes
: 
Z
VarIsInitializedOp_10VarIsInitializedOpSoftmax/kernel/Adam_1*
_output_shapes
: 
V
VarIsInitializedOp_11VarIsInitializedOpSoftmax/bias/Adam*
_output_shapes
: 
X
VarIsInitializedOp_12VarIsInitializedOpSoftmax/bias/Adam_1*
_output_shapes
: 
�
initNoOp^Conv1/bias/Adam/Assign^Conv1/bias/Adam_1/Assign^Conv1/bias/Assign^Conv1/kernel/Adam/Assign^Conv1/kernel/Adam_1/Assign^Conv1/kernel/Assign^Softmax/bias/Adam/Assign^Softmax/bias/Adam_1/Assign^Softmax/bias/Assign^Softmax/kernel/Adam/Assign^Softmax/kernel/Adam_1/Assign^Softmax/kernel/Assign^TFOptimizer/iterations/Assign(^training/TFOptimizer/beta1_power/Assign(^training/TFOptimizer/beta2_power/Assign
0

group_depsNoOp	^loss/mul^metrics/acc/Mean
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_6c3f1da70ca54a959dfa6aedd68f3ed1/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*�
value�B�B
Conv1/biasBConv1/bias/AdamBConv1/bias/Adam_1BConv1/kernelBConv1/kernel/AdamBConv1/kernel/Adam_1BSoftmax/biasBSoftmax/bias/AdamBSoftmax/bias/Adam_1BSoftmax/kernelBSoftmax/kernel/AdamBSoftmax/kernel/Adam_1BTFOptimizer/iterationsB training/TFOptimizer/beta1_powerB training/TFOptimizer/beta2_power*
dtype0*
_output_shapes
:
�
save/SaveV2/shape_and_slicesConst*1
value(B&B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesConv1/bias/Read/ReadVariableOp#Conv1/bias/Adam/Read/ReadVariableOp%Conv1/bias/Adam_1/Read/ReadVariableOp Conv1/kernel/Read/ReadVariableOp%Conv1/kernel/Adam/Read/ReadVariableOp'Conv1/kernel/Adam_1/Read/ReadVariableOp Softmax/bias/Read/ReadVariableOp%Softmax/bias/Adam/Read/ReadVariableOp'Softmax/bias/Adam_1/Read/ReadVariableOp"Softmax/kernel/Read/ReadVariableOp'Softmax/kernel/Adam/Read/ReadVariableOp)Softmax/kernel/Adam_1/Read/ReadVariableOp*TFOptimizer/iterations/Read/ReadVariableOp training/TFOptimizer/beta1_power training/TFOptimizer/beta2_power*
dtypes
2	
�
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*
_output_shapes
: *'
_class
loc:@save/ShardedFilename
�
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*

axis *
T0*
N*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst*�
value�B�B
Conv1/biasBConv1/bias/AdamBConv1/bias/Adam_1BConv1/kernelBConv1/kernel/AdamBConv1/kernel/Adam_1BSoftmax/biasBSoftmax/bias/AdamBSoftmax/bias/Adam_1BSoftmax/kernelBSoftmax/kernel/AdamBSoftmax/kernel/Adam_1BTFOptimizer/iterationsB training/TFOptimizer/beta1_powerB training/TFOptimizer/beta2_power*
dtype0*
_output_shapes
:
�
save/RestoreV2/shape_and_slicesConst*1
value(B&B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2	*P
_output_shapes>
<:::::::::::::::
N
save/Identity_1Identitysave/RestoreV2*
T0*
_output_shapes
:
S
save/AssignVariableOpAssignVariableOp
Conv1/biassave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:1*
T0*
_output_shapes
:
Z
save/AssignVariableOp_1AssignVariableOpConv1/bias/Adamsave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:2*
T0*
_output_shapes
:
\
save/AssignVariableOp_2AssignVariableOpConv1/bias/Adam_1save/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:3*
T0*
_output_shapes
:
W
save/AssignVariableOp_3AssignVariableOpConv1/kernelsave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:4*
T0*
_output_shapes
:
\
save/AssignVariableOp_4AssignVariableOpConv1/kernel/Adamsave/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:5*
T0*
_output_shapes
:
^
save/AssignVariableOp_5AssignVariableOpConv1/kernel/Adam_1save/Identity_6*
dtype0
P
save/Identity_7Identitysave/RestoreV2:6*
T0*
_output_shapes
:
W
save/AssignVariableOp_6AssignVariableOpSoftmax/biassave/Identity_7*
dtype0
P
save/Identity_8Identitysave/RestoreV2:7*
T0*
_output_shapes
:
\
save/AssignVariableOp_7AssignVariableOpSoftmax/bias/Adamsave/Identity_8*
dtype0
P
save/Identity_9Identitysave/RestoreV2:8*
T0*
_output_shapes
:
^
save/AssignVariableOp_8AssignVariableOpSoftmax/bias/Adam_1save/Identity_9*
dtype0
Q
save/Identity_10Identitysave/RestoreV2:9*
T0*
_output_shapes
:
Z
save/AssignVariableOp_9AssignVariableOpSoftmax/kernelsave/Identity_10*
dtype0
R
save/Identity_11Identitysave/RestoreV2:10*
T0*
_output_shapes
:
`
save/AssignVariableOp_10AssignVariableOpSoftmax/kernel/Adamsave/Identity_11*
dtype0
R
save/Identity_12Identitysave/RestoreV2:11*
T0*
_output_shapes
:
b
save/AssignVariableOp_11AssignVariableOpSoftmax/kernel/Adam_1save/Identity_12*
dtype0
R
save/Identity_13Identitysave/RestoreV2:12*
T0	*
_output_shapes
:
c
save/AssignVariableOp_12AssignVariableOpTFOptimizer/iterationssave/Identity_13*
dtype0	
�
save/AssignAssign training/TFOptimizer/beta1_powersave/RestoreV2:13*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*;
_class1
/-loc:@training/TFOptimizer/Read/ReadVariableOp
�
save/Assign_1Assign training/TFOptimizer/beta2_powersave/RestoreV2:14*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*=
_class3
1/loc:@training/TFOptimizer/Read_1/ReadVariableOp
�
save/restore_shardNoOp^save/Assign^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9^save/Assign_1
-
save/restore_allNoOp^save/restore_shard "<
save/Const:0save/Identity:0save/restore_all (5 @F8"�
trainable_variables��
x
Conv1/kernel:0Conv1/kernel/Assign"Conv1/kernel/Read/ReadVariableOp:0(2)Conv1/kernel/Initializer/random_uniform:08
g
Conv1/bias:0Conv1/bias/Assign Conv1/bias/Read/ReadVariableOp:0(2Conv1/bias/Initializer/zeros:08
�
Softmax/kernel:0Softmax/kernel/Assign$Softmax/kernel/Read/ReadVariableOp:0(2+Softmax/kernel/Initializer/random_uniform:08
o
Softmax/bias:0Softmax/bias/Assign"Softmax/bias/Read/ReadVariableOp:0(2 Softmax/bias/Initializer/zeros:08
�
TFOptimizer/iterations:0TFOptimizer/iterations/Assign,TFOptimizer/iterations/Read/ReadVariableOp:0(22TFOptimizer/iterations/Initializer/initial_value:08")
train_op

training/TFOptimizer/Adam"�
	variables��
x
Conv1/kernel:0Conv1/kernel/Assign"Conv1/kernel/Read/ReadVariableOp:0(2)Conv1/kernel/Initializer/random_uniform:08
g
Conv1/bias:0Conv1/bias/Assign Conv1/bias/Read/ReadVariableOp:0(2Conv1/bias/Initializer/zeros:08
�
Softmax/kernel:0Softmax/kernel/Assign$Softmax/kernel/Read/ReadVariableOp:0(2+Softmax/kernel/Initializer/random_uniform:08
o
Softmax/bias:0Softmax/bias/Assign"Softmax/bias/Read/ReadVariableOp:0(2 Softmax/bias/Initializer/zeros:08
�
TFOptimizer/iterations:0TFOptimizer/iterations/Assign,TFOptimizer/iterations/Read/ReadVariableOp:0(22TFOptimizer/iterations/Initializer/initial_value:08
�
"training/TFOptimizer/beta1_power:0'training/TFOptimizer/beta1_power/Assign'training/TFOptimizer/beta1_power/read:020training/TFOptimizer/beta1_power/initial_value:0
�
"training/TFOptimizer/beta2_power:0'training/TFOptimizer/beta2_power/Assign'training/TFOptimizer/beta2_power/read:020training/TFOptimizer/beta2_power/initial_value:0
�
Conv1/kernel/Adam:0Conv1/kernel/Adam/Assign'Conv1/kernel/Adam/Read/ReadVariableOp:0(2%Conv1/kernel/Adam/Initializer/zeros:0
�
Conv1/kernel/Adam_1:0Conv1/kernel/Adam_1/Assign)Conv1/kernel/Adam_1/Read/ReadVariableOp:0(2'Conv1/kernel/Adam_1/Initializer/zeros:0
y
Conv1/bias/Adam:0Conv1/bias/Adam/Assign%Conv1/bias/Adam/Read/ReadVariableOp:0(2#Conv1/bias/Adam/Initializer/zeros:0
�
Conv1/bias/Adam_1:0Conv1/bias/Adam_1/Assign'Conv1/bias/Adam_1/Read/ReadVariableOp:0(2%Conv1/bias/Adam_1/Initializer/zeros:0
�
Softmax/kernel/Adam:0Softmax/kernel/Adam/Assign)Softmax/kernel/Adam/Read/ReadVariableOp:0(2'Softmax/kernel/Adam/Initializer/zeros:0
�
Softmax/kernel/Adam_1:0Softmax/kernel/Adam_1/Assign+Softmax/kernel/Adam_1/Read/ReadVariableOp:0(2)Softmax/kernel/Adam_1/Initializer/zeros:0
�
Softmax/bias/Adam:0Softmax/bias/Adam/Assign'Softmax/bias/Adam/Read/ReadVariableOp:0(2%Softmax/bias/Adam/Initializer/zeros:0
�
Softmax/bias/Adam_1:0Softmax/bias/Adam_1/Assign)Softmax/bias/Adam_1/Read/ReadVariableOp:0(2'Softmax/bias/Adam_1/Initializer/zeros:0*�
serving_default�
;
input_image,
Conv1_input:0���������=
Softmax/Softmax:0(
Softmax/Softmax:0���������
tensorflow/serving/predict