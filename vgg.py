import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG(nn.Module):
	def __init__(self, features, out_fcl_1=4096, out_fcl_2 = 4096, num_classes=50, init_weights=True):
		super(VGG, self).__init__()
		self.features = features
		self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
		self.hl_1 = nn.Linear(512 * 7 * 7, out_fcl_1)
		self.hl_2 = nn.Linear(out_fcl_1, out_fcl_2)
		self.out_layer = nn.Linear(out_fcl_2, num_classes)
		self.dropout = nn.Dropout(0.5)

		if init_weights:
			self._initialize_weights()

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		h1_out = self.dropout(F.relu(self.hl_1(x)))
		h2_out = self.dropout(F.relu(self.hl_2(h1_out)))
		#h1_out = F.relu(self.hl_1(x))
		#h2_out = F.relu(self.hl_2(h1_out))

		out = self.out_layer(h2_out)
		#h1_out = self.hidden_layer_1(x)
		#h2_out = self.hidden_layer_2(h1_out)
		#out = self.out_layer(h2_out)
		#x = self.classifier(x)
		#print ('OUT:: {}'.format(out.size()))
		return x, h1_out, h2_out, out

	def forward_hidden_test(self, h_out, n_layer):
		out = []
		if (n_layer == -1):
			#h1_out = self.dropout(F.relu(self.hl_1(h_out)))
			#h2_out = self.dropout(F.relu(self.hl_2(h1_out)))
			h1_out = F.relu(self.hl_1(h_out))
			h2_out = F.relu(self.hl_2(h1_out))
			out = self.out_layer(h2_out)
		elif (n_layer == 0):
			# h2_out = self.dropout(F.relu(self.hl_2(h_out)))
			h2_out = F.relu(self.hl_2(h_out))
			out = self.out_layer(h2_out)
		elif (n_layer == 1):
			out = self.out_layer(h_out)
		else:
			x = self.features(h_out)
			x = self.avgpool(x)
			x = torch.flatten(x, 1)
			# h1_out = self.dropout(F.relu(self.hl_1(x)))
			# h2_out = self.dropout(F.relu(self.hl_2(h1_out)))
			h1_out = F.relu(self.hl_1(x))
			h2_out = F.relu(self.hl_2(h1_out))

			out = self.out_layer(h2_out)
		return out

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)