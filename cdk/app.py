#!/usr/bin/env python3
import aws_cdk as cdk
from hpv_mlops_stack import HpvMlopsStack

app = cdk.App()
HpvMlopsStack(app, "HpvMlopsStackV2", env=cdk.Environment(region="us-east-1"))
app.synth()