package com.riftboundjudge

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.navigation.compose.rememberNavController
import com.riftboundjudge.ui.AppNavGraph
import com.riftboundjudge.ui.modelgate.ModelGate
import com.riftboundjudge.ui.theme.RiftboundJudgeTheme
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            RiftboundJudgeTheme {
                ModelGate {
                    val nav = rememberNavController()
                    AppNavGraph(nav = nav)
                }
            }
        }
    }
}