[![](https://jitpack.io/v/AJ-1-code/NumJ.svg)](https://jitpack.io/#AJ-1-code/NumJ)

use this for build.gradle in dependenices: implementation 'com.github.AJ-1-code:NumJ:1.0.0'

make sure this is in your settings.gradle: dependencyResolutionManagement {
		repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
		repositories {
			mavenCentral()
			maven { url 'https://jitpack.io' }
		}
	}
