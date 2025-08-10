from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from predict import run_inference

