
import torch
import math

class SubsampledTRCGParams:
	def __init__( self ): 
		self.delta = 0
		self.max_delta = 0
		self.eta1 = 0
		self.eta2 = 0
		self.gamma1 = 0
		self.gamma2 = 0
		self.max_props = 0
		self.max_mvp = 0
		self.max_iters = 0
		self.sampleSize = 0
		self.min_delta = 0

class SubsampledTRCGStatistics : 
	def __init__ (self, logfile, console=False): 
		
		self.train_ll = 0
		self.test_ll = 0

		self.train_accu = 0
		self.test_accu = 0

		self.gradnorm = 0
		self.delta = 0

		self.tr_failures = 0
		self.cg_iterations = 0

		self.iteration_time = 0
		self.total_time = 0

		self.no_props = 0
		self.no_mvp = 0

		self.console = console

		self.out = open( logfile, 'w' )

	def printHeader(self): 
		self.out.write( "Tr.LL\tTr.Accu\tTest.LL\tTest.Acc\tgrad.norm\tdelta\t\tNo.Fails\tCG.Iters\tIter.Time\tTotal.Time\n")

	def printIterationStats( self ): 
		self.out.write( "%e\t%3.2f\t%e\t%3.2f\t%e\t%e\t%d\t%d\t%.3f\t%.3f\n" % (self.train_ll, math.fabs( self.train_accu ), self.test_ll, math.fabs( self.test_accu ), self.gradnorm, self.delta, self.tr_failures, self.cg_iterations, self.iteration_time, self.total_time) )

		if self.console: 
			print( "%e\t%3.2f\t%e\t%3.2f\t%e\t%e\t%d\t%d\t%.3f\t%.3f\n" % (self.train_ll, math.fabs( self.train_accu ), self.test_ll, math.fabs( self.test_accu ), self.gradnorm, self.delta, self.tr_failures, self.cg_iterations, self.iteration_time, self.total_time) )

		self.out.flush ()

	def shutdown( self ): 
		self.out.close ()

		
