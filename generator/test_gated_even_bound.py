"""Independent exact rational toy identity and Decimal norm/inner references."""
import ast
from decimal import Decimal,localcontext
from fractions import Fraction as F
from pathlib import Path
import unittest
import numpy as np
from gated_even_bound import Bounds,map_norms,row_record,summarize


def exact_dot(a,b):return sum((F(float(x))*F(float(y)) for x,y in zip(a,b)),F(0))


def dec(x):return Decimal(x.numerator)/Decimal(x.denominator)


class GatedEvenBoundTests(unittest.TestCase):
    def test_exact_scalar_reference(self):
        # Supplied dyadic products, a selected-column hostile residue and large cancellation.
        n,m,d=5,8,3;support=np.array([1,6]);rng=np.random.default_rng(79)
        products={name:rng.integers(-32,33,size=(1,n,m)).astype(float)/16 for name in ['L','T','H','F','c1','c2']}
        products['H']=-products['T']+products['H']/32;products['F']/=8192;products['c1']/=32768;products['c2']/=65536
        products['B']=products['T']+products['H']+products['F']+products['c2']
        products['R']=products['L']+products['B']+products['c1']
        down=rng.integers(-16,17,size=(d,m)).astype(float)/16;gate=np.array([.5,-.75,.25])
        def contract(a):return np.array([[[float(exact_dot(a[0,i],down[j])) for j in range(d)] for i in range(n)]])
        fields={name:products[name] for name in ['R','L','B','T','H','F']};fields.update(gate=gate.reshape(1,1,d))
        fields.update(YR=contract(fields['R']),YL=contract(fields['L']),YT=contract(fields['T']))
        fields['closure78']=fields['B']-((fields['T']+fields['H'])+fields['F'])
        fields['remaining']=(fields['YR']-fields['YL'])-fields['YT'];fields['gatedRemaining']=gate*fields['remaining'];fields['gatedQuadratic']=gate*fields['YT']
        bounds=Bounds();omitted,maps=map_norms(down,gate,support,bounds);rows=[]
        with localcontext() as ctx:
            ctx.prec=90
            for i in range(n):
                r=row_record(fields,i,omitted,maps,bounds);rows.append(r)
                yh=[exact_dot(fields['H'][0,i,omitted],down[j,omitted])*F(float(gate[j])) for j in range(d)]
                actual=[F(float(x))-y for x,y in zip(fields['gatedRemaining'][0,i],yh)]
                actualnorm=dec(sum((x*x for x in actual),F(0))).sqrt()
                self.assertLessEqual(actualnorm,Decimal(r['etaUpper']))
                hnorm=dec(sum((x*x for x in yh),F(0))).sqrt();lo,hi=r['evenOutputNormInterval']
                self.assertLessEqual(Decimal(lo),hnorm);self.assertGreaterEqual(Decimal(hi),hnorm)
                alignment=dec(sum((F(float(x))*y for x,y in zip(fields['gatedQuadratic'][0,i],yh)),F(0)))
                lo,hi=r['quadraticEvenInnerInterval'];self.assertLessEqual(Decimal(lo),alignment);self.assertGreaterEqual(Decimal(hi),alignment)
                self.assertGreater(sum(r['selectedStoredNonzeros'].values()),0)
            aggregate=summarize(rows,bounds)
            self.assertEqual(aggregate['rows'],n);self.assertTrue(bounds.status()['normalityScanApplicable'])

    def test_zero_and_subnormal_diagnostic(self):
        b=Bounds();self.assertEqual(b.norm(np.zeros(8)),[0,0]);b.norm([np.nextafter(0.0,1.0)])
        self.assertFalse(b.status()['normalityScanApplicable'])

    def test_rounded_contractions_and_changed_saved_grouping(self):
        rng=np.random.default_rng(7901);n,m,d=3,7,4;support=np.array([0,4]);gate=rng.normal(size=d)
        f={name:rng.normal(size=(1,n,m)) for name in ['R','L','B','T','H','F']}
        down=rng.normal(size=(d,m));f['gate']=gate.reshape(1,1,d)
        def contract(a):return np.array([[[float(exact_dot(a[0,i],down[j])) for j in range(d)] for i in range(n)]])
        f.update(YR=contract(f['R']),YL=contract(f['L']),YT=contract(f['T']))
        f['closure78']=f['B']-((f['T']+f['H'])+f['F']);f['remaining']=(f['YR']-f['YL'])-f['YT']
        f['remaining'][0,1,2]+=.125;f['gatedRemaining']=f['remaining']*gate;f['gatedQuadratic']=f['YT']*gate
        b=Bounds();omitted,maps=map_norms(down,gate,support,b)
        with localcontext() as ctx:
            ctx.prec=90
            for i in range(n):
                row=row_record(f,i,omitted,maps,b)
                yh=[exact_dot(f['H'][0,i,omitted],down[j,omitted])*F(float(gate[j])) for j in range(d)]
                error=[F(float(x))-h for x,h in zip(f['gatedRemaining'][0,i],yh)]
                self.assertLessEqual(dec(sum((e*e for e in error),F(0))).sqrt(),Decimal(row['etaUpper']))
                if i==1:self.assertGreater(row['components']['groupingMismatch'],0)

    def test_no_contraction_or_neural_call_in_bound_source(self):
        tree=ast.parse(Path(__file__).with_name('gated_even_bound.py').read_text())
        forbidden={'matmul','dot','einsum','tensordot','svd','silu','tanh','linear'}
        self.assertFalse(any(isinstance(x,ast.MatMult) for x in ast.walk(tree)))
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                self.assertNotIn(node.func.attr,forbidden-{'dot'})
                if node.func.attr=='dot':self.assertTrue(isinstance(node.func.value,ast.Name) and node.func.value.id=='bounds')
        # Bounds.dot is an elementwise product plus scalar reduction, not matrix multiplication.


if __name__=='__main__':unittest.main()
