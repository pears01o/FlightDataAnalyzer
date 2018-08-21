 node('bose') {
   stage('Checkout') {

        checkout scm 
    }
    try {
    stage('Run the NoseTests') {
          sh '''#!/bin/bash -l
          echo "######NoseTest #######"·
          docker-compose up
          '''
          junit 'nosetests.xml'
      }
    } catch (e) {
      notifyFailed()
      throw e;
      }
}
