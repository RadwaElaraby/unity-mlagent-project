using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Wall : MonoBehaviour
{
    private void OnCollisionEnter(Collision collision)
    {
        //if (collision.collider.CompareTag("agent"))
        //{
            //var agentObj = collision.collider.GetComponent<MyAgent>();
            //agentObj.SetReward(-2 / agentObj.MaxStep);
        //}
    }

}
