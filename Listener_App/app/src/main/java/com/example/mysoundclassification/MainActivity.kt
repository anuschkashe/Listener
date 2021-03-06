package com.example.mysoundclassification

import android.Manifest
import android.media.MediaRecorder
import android.os.Bundle
import android.os.CountDownTimer
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import org.tensorflow.lite.task.audio.classifier.AudioClassifier
import org.w3c.dom.Text
import java.util.*
import kotlin.concurrent.scheduleAtFixedRate


class MainActivity : AppCompatActivity() {
    var TAG = "MainActivity"

    var modelPath = "try9.tflite"
    var probabilityThreshold: Float = 0.4f

    private var mytimer: Timer? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val btnRecord = findViewById<Button>(R.id.btnRecord)
        val timeLimit = findViewById<TextView>(R.id.timeLimit)

        val REQUEST_RECORD_AUDIO = 1337
        requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_RECORD_AUDIO)

        var textViewClass = findViewById<TextView>(R.id.classout)
        var textViewProb = findViewById<TextView>(R.id.probout)
        val recorderSpecsTextView = findViewById<TextView>(R.id.textViewAudioRecorderSpecs)

        btnRecord.setOnClickListener {
            if (btnRecord.text.toString() == "Record and Classify") {

                object : CountDownTimer(2000, 500) {
                    override fun onTick(millisUntilFinished: Long) {
                        timeLimit.setText("seconds remaining: " + millisUntilFinished / 1000)
                    }

                    override fun onFinish() {
                        timeLimit.setText("done!")
                        mytimer?.cancel()
                        btnRecord.text = "Record and Classify"
                    }
                }.start()
                val classifier = AudioClassifier.createFromFile(this, modelPath)
                val tensor = classifier.createInputTensorAudio()
                val format = classifier.requiredTensorAudioFormat
                val recorderSpecs = "Number Of Channels: ${format.channels}\n" +
                        "Sample Rate: ${format.sampleRate}"
                recorderSpecsTextView.text = recorderSpecs

                val record = classifier.createAudioRecord()
                record.startRecording()

                mytimer = Timer().apply {
                    scheduleAtFixedRate(1, 109) {

                        tensor.load(record)
                        val output = classifier.classify(tensor)
                        val filteredModelOutput = output[0].categories.filter {
                            it.score > probabilityThreshold
                        }

                        val outputStr =
                            filteredModelOutput.sortedBy { -it.score }
                                .joinToString(separator = "\n") { "${it.label} -> ${it.score} " }

                        val outputClass =
                            filteredModelOutput.sortedBy { -it.score }
                                .joinToString(separator = "\n") { "${it.label}" }

                        val outputProb =
                            filteredModelOutput.sortedBy { -it.score }
                                .joinToString(separator = "\n") { "${it.score}" }

                        val outP = outputProb.take(4)

                        if (outputStr.isNotEmpty())
                            runOnUiThread {
                                textViewClass.text = outputClass
                                textViewProb.text = outP
                            }
                    }
                }
                btnRecord.text = "Say something"
            }
        }
    }
}